use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::Write;
use std::{fmt, io};
use std::fmt::{Debug, Formatter, write};

use crate::type_checking::{ExprRef, ExprToType, Expression, Item};
use crate::types::{Type, TypeRef};
use crate::{ast, Arena};

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub(crate) struct Var(usize);

pub(crate) fn make_expr<'expr, 'tcx>(
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
    ty: TypeRef<'tcx>,
    expr: Expression<'expr>,
) -> ExprRef<'expr> {
    let expr = exprs.alloc(expr);
    type_by_expr.insert(expr, ty);
    expr
}

impl Var {
    fn error() -> Self {
        Self(usize::MAX)
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.0 == usize::MAX {
            write!(f, "<error>>")
        } else {
            write!(f, "_{}", self.0)
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Signedness {
    Unsigned,
    Signed,
    Unspecified,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Bits {
    value: u64,
    width: u32,
    sign: Signedness,
}

impl Debug for Bits {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)?;
        write!(f, "_{}", match self.sign {
            Signedness::Unsigned => "u",
            Signedness::Signed => "i",
            Signedness::Unspecified => "?"
        })?;
        write!(f, "{}", self.width)?;
        Ok(())
    }
}

macro_rules! impl_bits_for {
    ($t: ty) => {
        impl From<$t> for Bits {
            #[inline]
            fn from(value: $t) -> Self {
                Self {
                    value: value as u64,
                    width: <$t>::BITS,
                    sign: if <$t>::MIN == 0 { Signedness::Unsigned } else { Signedness::Signed },
                }
            }
        }
    };
}

impl_bits_for!(usize);
impl_bits_for!(i32);
impl_bits_for!(u32);
impl_bits_for!(i64);

impl Bits {
    fn value(&self) -> i64 {
        let mask = ((1u128 << (self.width + 1)) - 1) as u64;
        let sign = 1u64 << (self.width - 1);
        if self.value & mask & sign != 0 {
            let sign_ext = u64::MAX ^ mask;
            ((self.value & mask) | sign_ext) as _
        } else {
            (self.value & mask) as _
        }
    }

    #[track_caller]
    fn check_compatible(&self, other: &Self) {
        if self.width != other.width || self.sign != other.sign {
            log::warn!("incompatible types: {:?} {:?}", self, other);
        }
    }

    fn add(&self, other: &Self) -> Self {
        self.check_compatible(other);
        Self {
            value: (self.value() + other.value()) as _,
            width: self.width,
            sign: self.sign,
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self.check_compatible(other);
        Self {
            value: self.value().saturating_sub(other.value()) as _,
            width: self.width,
            sign: self.sign,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        self.check_compatible(other);
        Self {
            value: (self.value() * other.value()) as _,
            width: self.width,
            sign: self.sign,
        }
    }

    fn div(&self, other: &Self) -> Self {
        self.check_compatible(other);
        Self {
            value: (self.value() / other.value()) as _,
            width: self.width,
            sign: self.sign,
        }
    }

    fn cmp(&self, other: &Self) -> Ordering {
        self.check_compatible(other);
        self.value().cmp(&other.value())
    }

    fn negate(&self) -> Self {
        assert_ne!(self.sign, Signedness::Unsigned);
        Self {
            value: (-self.value()) as _,
            width: self.width,
            sign: self.sign,
        }
    }

    fn as_usize(&self) -> usize {
        self.value.try_into().unwrap()
    }

    fn as_i32(&self) -> i32 {
        self.value as i32
    }

    fn as_u32(&self) -> u32 {
        self.value as u32
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::Bits;

    #[test]
    fn test_bits_value() {
        let a = Bits::from(-10i32);
        assert_eq!(a.value(), -10);
        assert_eq!(a.negate().value(), 10);
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum Const {
    Integer(Bits),
    F32(f32),
    Bool(bool),
    Pointer(usize),
    Struct,
    Undefined,
}

impl Expression<'_> {
    pub(crate) fn as_const(&self) -> Option<Const> {
        match self {
            Expression::Integer(bits) => Some(Const::Integer(*bits)),
            Expression::Float(x) => Some(Const::F32(*x as _)),
            Expression::Prefix(ast::Operator::Negate, inner) => match inner.as_const()? {
                Const::Integer(val) => Some(Const::Integer(val.negate())),
                _ => None,
            },
            Expression::Bool(x) => Some(Const::Bool(*x)),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum CastType {
    F32ToI32,
    I32ToF32,
    U32ToF32,
    U32ToI32,
    I32ToU32,
}

enum Instr {
    Const(Var, Const),
    Copy(Var, Var),
    UnaryOperation(Var, ast::Operator, Var),
    BinaryOperation(Var, ast::Operator, Var, Var),
    SetElement(Var, usize, Var),
    GetElement(Var, Var, Var),
    SetField(Var, usize, Var),
    GetField(Var, Var, usize),
    Call(Var, String, Vec<Var>),
    Cast(Var, Var, CastType),
}

impl fmt::Debug for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Instr::Const(var, val) => write!(f, "{:?} = {:?}", var, val),
            Instr::Copy(lhs, rhs) => write!(f, "{:?} = {:?}", lhs, rhs),
            Instr::BinaryOperation(left, op, a, b) => write!(
                f,
                "{:?} = {:?} {} {:?}",
                left,
                a,
                match op {
                    ast::Operator::Add => "+",
                    ast::Operator::Sub => "-",
                    ast::Operator::Mul => "*",
                    ast::Operator::Div => "/",
                    ast::Operator::Equal => "==",
                    ast::Operator::NotEqual => "!=",
                    ast::Operator::Less => "<",
                    ast::Operator::LessEqual => "<=",
                    ast::Operator::Greater => ">",
                    ast::Operator::GreaterEqual => ">=",
                    ast::Operator::And => "and",
                    ast::Operator::Or => "or",
                    _ => unimplemented!(),
                },
                b
            ),
            Instr::UnaryOperation(left, op, a) => write!(
                f,
                "{:?} = {}{:?}",
                left,
                match op {
                    ast::Operator::Negate => "-",
                    ast::Operator::Deref => "*",
                    ast::Operator::Ref => "&",
                    _ => unimplemented!("{:?}", op),
                },
                a
            ),
            Instr::SetElement(arr, index, val) => {
                write!(f, "{:?}[{}] = {:?}", arr, index, val)
            }
            Instr::GetElement(var, arr, index) => {
                write!(f, "{:?} = {:?}[{:?}]", var, arr, index)
            }
            Instr::Cast(left, right, mode) => {
                write!(f, "{:?} = cast({:?}, {:?})", left, right, mode)
            }
            Instr::Call(target, ident, args) => {
                write!(f, "{:?} = {}({:?})", target, ident, args)
            }
            Instr::SetField(target, field, value) => {
                write!(f, "{:?}.{} = {:?}", target, field, value)
            }
            Instr::GetField(var, base, value) => {
                write!(f, "{:?} = {:?}.{}", var, base, value)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Block(usize);

#[derive(Debug)]
enum Terminator {
    Jump(Block),
    JumpIf(Var, Block, Block),
    Return,
    Unreachable,
    Assert(Var, Block),
}

#[derive(Debug)]
struct VarDef<'tcx> {
    name: Option<String>,
    ty: TypeRef<'tcx>,
}

#[derive(Debug)]
struct BlockBody {
    instrs: Vec<Instr>,
    terminator: Terminator,
}

#[derive(Debug)]
pub(crate) struct FunctionIr<'tcx> {
    name: String,
    num_args: usize,
    /// defines[0..num_args] == function args
    ///
    /// defines[num_args] == function return types
    ///
    /// defines[num_args+1..] == locals
    defines: Vec<VarDef<'tcx>>,
    blocks: Vec<BlockBody>,
}

pub(crate) fn validate_types(ir: &FunctionIr<'_>) {
    for x in &ir.defines {
        match &x.ty {
            Type::Integer | Type::Float => {
                log::warn!("unexpected abstract type {:?} for {:?}", x.ty, &x.name);
            }
            _ => {}
        }
    }
}

pub(crate) fn dump_ir(ir: &FunctionIr<'_>, f: &mut impl Write) -> io::Result<()> {
    write!(f, "fn {}(", ir.name)?;
    for (idx, it) in ir.defines.iter().enumerate().take(ir.num_args) {
        if idx != 0 {
            write!(f, ", ")?;
        }
        write!(f, "_{}: {:?}", idx, it.ty)?;
    }
    writeln!(f, ") -> {:?} {{", ir.defines[ir.num_args].ty)?;

    for (idx, it) in ir.defines.iter().enumerate().skip(ir.num_args + 1) {
        writeln!(
            f,
            "  let _{}: {:?}; // {}",
            idx,
            it.ty,
            &it.name.as_deref().unwrap_or("")
        )?;
    }

    for (idx, block) in ir.blocks.iter().enumerate() {
        writeln!(f, "  _bb{} {{", idx)?;
        for inst in &block.instrs {
            writeln!(f, "    {:?};", inst)?;
        }
        match block.terminator {
            Terminator::Jump(b) => writeln!(f, "    Jump(_bb{});", b.0)?,
            Terminator::JumpIf(v, bt, bf) => {
                writeln!(f, "    JumpIf(_{}, _bb{}, _bb{});", v.0, bt.0, bf.0)?
            }
            Terminator::Return => writeln!(f, "    Return;")?,
            Terminator::Unreachable => writeln!(f, "    Unreachable;")?,
            Terminator::Assert(v, next) => writeln!(f, "    Assert(_{}, _bb{});", v.0, next.0)?,
        }
        writeln!(f, "  }}")?;
    }

    writeln!(f, "}}")?;

    Ok(())
}

struct IrBuilder<'tcx> {
    args: usize,
    vars: Vec<VarDef<'tcx>>,
    blocks: Vec<BlockBody>,
}

impl<'tcx> IrBuilder<'tcx> {
    fn new() -> Self {
        Self {
            args: 0,
            vars: vec![],
            blocks: Default::default(),
        }
    }

    fn make_arg(&mut self, ty: TypeRef<'tcx>, name: Option<&str>) -> Var {
        assert_eq!(self.args, self.vars.len());
        self.args += 1;
        self.make_var(ty, name)
    }

    fn make_ret(&mut self, ty: TypeRef<'tcx>) -> Var {
        assert_eq!(self.args, self.vars.len());
        self.make_var(ty, None)
    }

    fn make_var(&mut self, ty: TypeRef<'tcx>, name: Option<&str>) -> Var {
        let var = Var(self.vars.len());
        self.vars.push(VarDef {
            ty,
            name: name.map(String::from),
        });
        var
    }

    fn build(self, name: String) -> FunctionIr<'tcx> {
        FunctionIr {
            name,
            num_args: self.args,
            defines: self.vars,
            blocks: self.blocks,
        }
    }

    fn block(&mut self) -> Block {
        let block = Block(self.blocks.len());
        self.blocks.push(BlockBody {
            instrs: vec![],
            terminator: Terminator::Unreachable,
        });
        block
    }

    fn set_terminator_of(&mut self, block: Block, term: Terminator) {
        self.blocks[block.0].terminator = term;
    }

    fn push(&mut self, block: Block, inst: Instr) {
        self.blocks[block.0].instrs.push(inst);
    }
}

fn visit_expr<'expr, 'tcx>(
    expr: ExprRef<'expr>,
    builder: &mut IrBuilder<'tcx>,
    names: &HashMap<String, Var>,
    block: Block,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> Var {
    match expr {
        Expression::Identifier(ident) => names[ident],
        Expression::Integer(val) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            builder.push(block, Instr::Const(var, Const::Integer(*val as _)));
            var
        }
        Expression::Float(val) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            builder.push(block, Instr::Const(var, Const::F32(*val as f32)));
            var
        }
        Expression::Bool(val) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            builder.push(block, Instr::Const(var, Const::Bool(*val)));
            var
        }
        Expression::Prefix(op, rhs) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            let operand = visit_expr(&rhs, builder, &names, block, exprs, type_by_expr);
            builder.push(block, Instr::UnaryOperation(var, *op, operand));
            var
        }
        Expression::Infix(op, lhs, rhs) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            let a = visit_expr(&lhs, builder, &names, block, exprs, type_by_expr);
            let b = visit_expr(&rhs, builder, &names, block, exprs, type_by_expr);
            builder.push(block, Instr::BinaryOperation(var, *op, a, b));
            var
        }
        Expression::Array(elements) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            for (index, item) in elements.iter().enumerate() {
                let expr = visit_expr(&item, builder, names, block, exprs, type_by_expr);
                builder.push(block, Instr::SetElement(var, index, expr));
            }
            var
        }
        Expression::Index(slice, index) => {
            let ty = match type_by_expr.of(slice) {
                Type::Array(_, ty) => ty,
                Type::Slice(_) => unimplemented!(),
                _ => unreachable!(),
            };

            let slice_var = match &slice {
                Expression::Identifier(ident) => names[ident],
                _ => unimplemented!(),
            };

            let element_var = builder.make_var(ty, None);
            let index_var = visit_expr(&index, builder, names, block, exprs, type_by_expr);
            builder.push(block, Instr::GetElement(element_var, slice_var, index_var));
            element_var
        }
        Expression::Call(func, args) => {
            let ident = match &func {
                Expression::Identifier(ident) => ident.clone(),
                Expression::Intrinsic(id) => format!("intrinsic.{}", id.to_str()),
                other => todo!("{:?}", other),
            };

            let mut params = Vec::new();
            for arg in args {
                params.push(visit_expr(&arg, builder, names, block, exprs, type_by_expr));
            }

            let ret = builder.make_var(type_by_expr.of(expr), Some("Return of"));
            builder.push(block, Instr::Call(ret, ident, params));
            ret
        }
        Expression::Range(_, _) => Var::error(),
        Expression::Var(var) => var.clone(),
        Expression::Error => Var::error(),
        Expression::Cast(source_expr) => {
            let var = builder.make_var(type_by_expr.of(source_expr), None);
            let x = visit_expr(&source_expr, builder, names, block, exprs, type_by_expr);
            builder.push(
                block,
                Instr::Cast(
                    var,
                    x,
                    match (type_by_expr.of(source_expr), type_by_expr.of(expr)) {
                        (Type::F32, Type::I32) => CastType::F32ToI32,
                        (Type::I32, Type::F32) => CastType::I32ToF32,
                        (Type::U32, Type::F32) => CastType::U32ToF32,
                        (Type::U32, Type::I32) => CastType::U32ToI32,
                        (Type::I32, Type::U32) => CastType::I32ToU32,
                        (a, b) => todo!("{:?} {:?}", a, b),
                    },
                ),
            );
            var
        }
        Expression::Intrinsic(_) => panic!(),
        Expression::Tuple(fields) | Expression::StructLiteral(fields) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            for (idx, field) in fields.iter().enumerate() {
                let x = visit_expr(&&field, builder, names, block, exprs, type_by_expr);
                builder.push(block, Instr::SetField(var, idx, x));
            }
            var
        }
        Expression::Field(expr, idx) => {
            let base = visit_expr(&&expr, builder, names, block, exprs, type_by_expr);
            let element_var = builder.make_var(type_by_expr.of(expr), None);
            builder.push(block, Instr::GetField(element_var, base, *idx));
            element_var
        }
    }
}

fn visit_item<'expr, 'tcx>(
    item: &Item<'expr, 'tcx>,
    arena: &'tcx Arena<Type<'tcx>>,
    builder: &mut IrBuilder<'tcx>,
    local_names: &mut HashMap<String, Var>,
    ret: Option<Var>,
    after_loop: Option<Block>,
    block: Block,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> Block {
    match item {
        Item::Let { name, ty, expr } => {
            let var = builder.make_var(ty, Some(name.as_str()));
            local_names.insert(name.clone(), var);

            if let Some(expr) = expr {
                let expr = visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
                builder.push(block, Instr::Copy(var, expr));
            }

            block
        }
        Item::Assignment {
            lhs,
            operator,
            expr,
        } => {
            let rhs = visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
            let lhs = visit_expr(lhs, builder, &local_names, block, exprs, type_by_expr);
            if let Some(op) = operator {
                builder.push(block, Instr::BinaryOperation(lhs, *op, lhs, rhs));
            } else {
                builder.push(block, Instr::Copy(lhs, rhs));
            }

            block
        }
        Item::Expression { expr } => {
            visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
            block
        }
        Item::If {
            condition,
            arm_true,
            arm_false,
        } => {
            let cond_var = visit_expr(
                &condition,
                builder,
                &local_names,
                block,
                exprs,
                type_by_expr,
            );

            let first_block_true = builder.block();
            let succ_block = builder.block();

            let mut block_true = first_block_true;
            builder.set_terminator_of(first_block_true, Terminator::Jump(succ_block));
            for item in arm_true {
                block_true = visit_item(
                    item,
                    arena,
                    builder,
                    local_names,
                    ret,
                    after_loop,
                    block_true,
                    exprs,
                    type_by_expr,
                );
            }

            let block_false = if let Some(items) = arm_false {
                let first_block_false = builder.block();

                let mut block_false = first_block_false;
                builder.set_terminator_of(block_false, Terminator::Jump(succ_block));
                for item in items {
                    block_false = visit_item(
                        item,
                        arena,
                        builder,
                        local_names,
                        ret,
                        after_loop,
                        block_false,
                        exprs,
                        type_by_expr,
                    );
                }

                first_block_false
            } else {
                succ_block
            };

            builder.set_terminator_of(
                block,
                Terminator::JumpIf(cond_var, first_block_true, block_false),
            );
            succ_block
        }
        Item::Return(expr) => {
            let var = visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
            builder.push(block, Instr::Copy(ret.unwrap(), var));
            builder.set_terminator_of(block, Terminator::Return);
            block
        }
        Item::ForIn { name, expr, body } => match type_by_expr.of(expr) {
            &Type::Array(len, item_ty) => {
                let items_id = String::from("_items");
                let index_id = String::from("_x");

                let expr_ty = type_by_expr.of(expr);
                let items = vec![
                    Item::Let {
                        name: items_id.clone(),
                        ty: expr_ty,
                        expr: Some(expr),
                    },
                    Item::Let {
                        name: index_id.clone(),
                        ty: arena.alloc(Type::I32),
                        expr: Some(make_expr(
                            exprs,
                            type_by_expr,
                            &Type::I32,
                            Expression::Integer(0.into()),
                        )),
                    },
                    Item::Loop {
                        body: {
                            let expr = Expression::Infix(
                                ast::Operator::Equal,
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Integer((len as u32).into()),
                                ),
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index_id.clone()),
                                ),
                            );

                            let e = Expression::Index(
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    expr_ty,
                                    Expression::Identifier(items_id),
                                ),
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::U32,
                                    Expression::Identifier(index_id.clone()),
                                ),
                            );

                            let mut items = vec![
                                Item::If {
                                    condition: make_expr(exprs, type_by_expr, &Type::Bool, expr),
                                    arm_true: vec![Item::Break],
                                    arm_false: None,
                                },
                                Item::Let {
                                    name: name.to_string(),
                                    ty: item_ty,
                                    expr: Some(make_expr(exprs, type_by_expr, item_ty, e)),
                                },
                            ];
                            items.extend_from_slice(body);
                            let e = Expression::Infix(
                                ast::Operator::Add,
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index_id.clone()),
                                ),
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Integer(1.into()),
                                ),
                            );
                            items.push(Item::Assignment {
                                lhs: make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index_id),
                                ),
                                operator: None,
                                expr: make_expr(exprs, type_by_expr, &Type::I32, e),
                            });
                            items
                        },
                    },
                ];

                visit_item(
                    &Item::Block(items),
                    arena,
                    builder,
                    local_names,
                    ret,
                    None,
                    block,
                    exprs,
                    type_by_expr,
                )
            }
            &Type::Range => {
                let Expression::Range(start_expr, end_expr) = &expr else {
                    todo!();
                };
                let end_expr = end_expr.as_ref().unwrap();

                let start = visit_expr(
                    &start_expr,
                    builder,
                    local_names,
                    block,
                    exprs,
                    type_by_expr,
                );
                let end = visit_expr(&end_expr, builder, local_names, block, exprs, type_by_expr);

                let index = name.clone();

                let items = vec![
                    Item::Let {
                        name: index.clone(),
                        ty: &Type::I32,
                        expr: Some(make_expr(
                            exprs,
                            type_by_expr,
                            &Type::I32,
                            Expression::Integer(0.into()),
                        )),
                    },
                    Item::Loop {
                        body: {
                            let e = Expression::Infix(
                                ast::Operator::Equal,
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index.clone()),
                                ),
                                make_expr(exprs, type_by_expr, &Type::I32, Expression::Var(end)),
                            );
                            let mut items = vec![Item::If {
                                condition: make_expr(exprs, type_by_expr, &Type::Bool, e),
                                arm_true: vec![Item::Break],
                                arm_false: None,
                            }];
                            items.extend_from_slice(body);
                            let e = Expression::Infix(
                                ast::Operator::Add,
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index.clone()),
                                ),
                                make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Integer(1.into()),
                                ),
                            );
                            items.push(Item::Assignment {
                                lhs: make_expr(
                                    exprs,
                                    type_by_expr,
                                    &Type::I32,
                                    Expression::Identifier(index),
                                ),
                                operator: None,
                                expr: make_expr(exprs, type_by_expr, &Type::I32, e),
                            });
                            items
                        },
                    },
                ];

                visit_item(
                    &Item::Block(items),
                    arena,
                    builder,
                    local_names,
                    ret,
                    None,
                    block,
                    exprs,
                    type_by_expr,
                )
            }
            other => {
                log::error!("Unsupported {:?}", other);
                block
            }
        },
        Item::Break => {
            builder.set_terminator_of(block, Terminator::Jump(after_loop.unwrap()));
            block
        }
        Item::Loop { body } => {
            let entry = builder.block();
            let mut current = entry;

            let after = builder.block();

            builder.set_terminator_of(block, Terminator::Jump(entry));
            for item in body {
                match item {
                    Item::Break => {}
                    Item::Yield(_) => {}
                    Item::Return(_) => {}
                    other => {
                        current = visit_item(
                            other,
                            arena,
                            builder,
                            local_names,
                            ret,
                            Some(after),
                            current,
                            exprs,
                            type_by_expr,
                        );
                        builder.set_terminator_of(current, Terminator::Jump(entry));
                    }
                }
            }
            builder.set_terminator_of(after, Terminator::Return);
            after
        }
        Item::Block(body) => {
            // FIXME: build a new block?
            let mut block = block;
            for item in body {
                block = visit_item(
                    item,
                    arena,
                    builder,
                    local_names,
                    ret,
                    after_loop,
                    block,
                    exprs,
                    type_by_expr,
                );
            }
            block
        }
        Item::Function { .. } => block,
        Item::Assert(expr) => block,
        other => unimplemented!("{:?}", other),
    }
}

pub(crate) fn build_ir<'expr, 'tcx>(
    item: &Item<'expr, 'tcx>,
    arena: &'tcx Arena<Type<'tcx>>,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> Result<FunctionIr<'tcx>, ()> {
    let mut builder = IrBuilder::new();
    match item {
        Item::Function {
            name,
            is_extern,
            args,
            ty,
            body,
        } if !is_extern => {
            let mut names = HashMap::new();
            for arg in args {
                let var = builder.make_arg(arg.ty, Some(name.as_str()));
                names.insert(arg.name.clone(), var);
            }
            let ret = builder.make_ret(ty);

            let mut block = builder.block();
            for item in body {
                block = visit_item(
                    item,
                    arena,
                    &mut builder,
                    &mut names,
                    Some(ret),
                    None,
                    block,
                    exprs,
                    type_by_expr,
                );
            }

            return Ok(builder.build(name.to_owned()));
        }
        _ => eprintln!("trying generate ir of item that is not a function"),
    }
    Err(())
}

#[derive(Default)]
struct ExecutionContext {
    vars: HashMap<Var, Const>,
    vars_arrays: HashMap<(Var, usize), Const>,
    memory: Vec<Const>,
}

impl ExecutionContext {
    fn insert_value(&mut self, dst: Var, val: Const) {
        if let Some(Const::Pointer(idx)) = self.vars.get(&dst) {
            self.memory[*idx] = val;
        } else {
            self.vars.insert(dst, val);
        }
    }
}

pub(crate) fn execute_ir(
    ir: &FunctionIr<'_>,
    args: &[Const],
    functions: &HashMap<String, FunctionIr<'_>>,
) -> Const {
    let mut curr_block = 0;
    let mut curr_inst = 0;

    let mut ctx = ExecutionContext::default();
    for (idx, cnst) in args.iter().enumerate().take(ir.num_args) {
       ctx.insert_value(Var(idx), *cnst);
    }

    loop {
        match ir.blocks[curr_block].instrs.get(curr_inst) {
            Some(instr) => {
                match instr {
                    Instr::Const(dst, val) => {
                        ctx.insert_value(*dst, *val);
                    }
                    Instr::Copy(dst, src) => {
                        match ctx.vars.get(src).copied() {
                            Some(var) => {
                                ctx.insert_value(*dst, var);

                                if let Const::Struct = var {
                                    // Copy all fields
                                    let mut idx = 0;
                                    while let Some(c) = ctx.vars_arrays.get(&(*src, idx)) {
                                        ctx.vars_arrays.insert((*dst, idx), c.clone());
                                        idx += 1;
                                    }
                                }
                            }
                            None => {
                                let mut to_copy = vec![];
                                for ((var, idx), _) in ctx.vars_arrays.iter() {
                                    if var == src {
                                        to_copy.push(*idx);
                                    }
                                }

                                for to in to_copy {
                                    ctx.vars_arrays.insert((*dst, to), ctx.vars_arrays[&(*src, to)]);
                                }
                            }
                        }
                    }
                    Instr::UnaryOperation(dst, op, a) => {
                        let val = match (op, ctx.vars[a]) {
                            (ast::Operator::Negate, Const::Integer(v)) => {
                                Const::Integer(v.negate())
                            }
                            (ast::Operator::Negate, Const::F32(v)) => Const::F32(-v),
                            (ast::Operator::Ref, val) => {
                                let idx = ctx.memory.len();
                                ctx.memory.push(val);
                                Const::Pointer(idx)
                            }
                            (ast::Operator::Deref, val) => val,
                            _ => {
                                unimplemented!("{:?} {:?}", op, a);
                            }
                        };
                        ctx.insert_value(*dst, val);
                    }
                    Instr::BinaryOperation(dst, op, a, b) => {
                        fn arithmetic_operation(op: ast::Operator, a: Const, b: Const) -> Option<Const> {
                            Some(match (op, a, b) {
                                (ast::Operator::And, Const::Bool(a), Const::Bool(b)) => {
                                    Const::Bool(a && b)
                                }
                                (ast::Operator::Or, Const::Bool(a), Const::Bool(b)) => {
                                    Const::Bool(a || b)
                                }
                                (ast::Operator::Add, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Integer(a.add(&b))
                                }
                                (ast::Operator::Mul, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Integer(a.mul(&b))
                                }
                                (ast::Operator::Sub, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Integer(a.sub(&b))
                                }
                                (ast::Operator::Div, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Integer(a.div(&b))
                                }
                                (ast::Operator::Less, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_lt())
                                }
                                (ast::Operator::Greater, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_gt())
                                }
                                (ast::Operator::Equal, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_eq())
                                }
                                (ast::Operator::NotEqual, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_ne())
                                }
                                (ast::Operator::LessEqual, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_le())
                                }
                                (ast::Operator::GreaterEqual, Const::Integer(a), Const::Integer(b)) => {
                                    Const::Bool(a.cmp(&b).is_ge())
                                }

                                (ast::Operator::Add, Const::F32(a), Const::F32(b)) => Const::F32(a + b),
                                (ast::Operator::Mul, Const::F32(a), Const::F32(b)) => Const::F32(a * b),
                                (ast::Operator::Sub, Const::F32(a), Const::F32(b)) => Const::F32(a - b),
                                (ast::Operator::Div, Const::F32(a), Const::F32(b)) => Const::F32(a / b),
                                (ast::Operator::Less, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a < b)
                                }
                                (ast::Operator::Greater, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a > b)
                                }
                                (ast::Operator::Equal, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a == b)
                                }
                                (ast::Operator::NotEqual, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a != b)
                                }
                                (ast::Operator::LessEqual, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a <= b)
                                }
                                (ast::Operator::GreaterEqual, Const::F32(a), Const::F32(b)) => {
                                    Const::Bool(a >= b)
                                }
                                _ => return None,
                            })
                        }

                        let a = ctx.vars.get(a).copied().unwrap_or(Const::Undefined);
                        let b = ctx.vars.get(b).copied().unwrap_or(Const::Undefined);
                        let val = match arithmetic_operation(*op, a, b) {
                            Some(x) => x,
                            None => match (op, a, b) {
                                (op, a, Const::Pointer(b)) => {
                                    arithmetic_operation(*op, a, ctx.memory[b]).unwrap_or(Const::Undefined)
                                }
                                (op, Const::Pointer(a), b) => {
                                    arithmetic_operation(*op, ctx.memory[a], b).unwrap_or(Const::Undefined)
                                }
                                (op, Const::Undefined, _) => {
                                    log::warn!("Propagating undefined value for {a:?} {op:?} {b:?} from {a:?}");
                                    Const::Undefined
                                }
                                (op, _, Const::Undefined) => {
                                    log::warn!("Propagating undefined value for {a:?} {op:?} {b:?} from {b:?}");
                                    Const::Undefined
                                }
                                (op, a, b) => {
                                    log::warn!("Missing operation for {a:?} {op:?} {b:?}");
                                    Const::Undefined
                                }
                            },
                        };
                        ctx.insert_value(*dst, val);
                    }
                    Instr::SetElement(arr, index, val) => {
                        ctx.vars_arrays.insert((*arr, *index), ctx.vars[val]);
                    }
                    Instr::GetElement(var, arr, index) => {
                        let index = match ctx.vars[index] {
                            Const::Integer(v) => v.as_usize(),
                            other => unimplemented!("{:?}", other),
                        };
                        ctx.insert_value(*var, ctx.vars_arrays[&(*arr, index)]);
                    }
                    Instr::Cast(target, source, mode) => {
                        let source = ctx.vars.get(source).copied().unwrap_or(Const::Undefined);
                        ctx.insert_value(
                            *target,
                            match (mode, source) {
                                (_, Const::Undefined) => Const::Undefined,
                                (CastType::F32ToI32, Const::F32(val)) => {
                                    Const::Integer((val as i32).into())
                                }
                                (CastType::I32ToF32, Const::Integer(val)) => {
                                    Const::F32(val.as_i32() as _)
                                }
                                (CastType::U32ToF32, Const::Integer(val)) => {
                                    Const::F32(val.as_u32() as _)
                                }
                                (CastType::U32ToI32, Const::Integer(val)) => {
                                    Const::Integer(val.try_into().unwrap())
                                }
                                _ => todo!("{:?} {:?}", mode, &source),
                            },
                        );
                    }
                    Instr::Call(target, name, args) => match name.as_str() {
                        "intrinsic.debug" => {
                            assert_eq!(args.len(), 1);
                            let value = ctx.vars[&args[0]];
                            log::debug!("Debug value: {:?}", &value);
                            ctx.insert_value(*target, value);
                        }
                        name => {
                            let func = &functions[name];
                            let args: Vec<_> = args.iter().map(|it| ctx.vars[it]).collect();
                            let result = execute_ir(func, &args, functions);
                            ctx.insert_value(*target, result);
                        }
                    },
                    Instr::SetField(lhs, idx, rhs) => {
                        ctx.vars_arrays.insert((*lhs, *idx), ctx.vars[rhs]);
                        ctx.insert_value(*lhs, Const::Struct);
                    }
                    Instr::GetField(target, lhs, idx) => {
                        ctx.insert_value(*target, ctx.vars_arrays.get(&(*lhs, *idx)).cloned().unwrap());
                    }
                }
                curr_inst += 1;
            }
            None => match ir.blocks[curr_block].terminator {
                Terminator::Jump(block) => {
                    curr_block = block.0;
                    curr_inst = 0;
                }
                Terminator::JumpIf(var, if_true, if_else) => match ctx.vars[&var] {
                    Const::Bool(v) => {
                        curr_block = if v { if_true.0 } else { if_else.0 };
                        curr_inst = 0;
                    }
                    Const::Undefined => {
                        log::error!("trying to jump to undefined");
                        return Const::Undefined;
                    }
                    other => unimplemented!("{:?}", other),
                },
                Terminator::Return => {
                    return match ctx.vars.get(&Var(ir.num_args)) {
                        Some(Const::Pointer(x)) => ctx.memory[*x],
                        Some(x) => *x,
                        None => Const::Undefined,
                    };
                }
                Terminator::Unreachable => {
                    log::warn!("executing unreachable");
                    return Const::Undefined;
                }
                Terminator::Assert(var, block) => match ctx.vars[&var] {
                    Const::Bool(true) => {
                        curr_block = block.0;
                        curr_inst = 0;
                    }
                    Const::Bool(false) => {
                        log::error!("Assertion failed!!!!");
                    }
                    other => unimplemented!("{:?}", other),
                },
            },
        }
    }
}
