use std::{fmt, io};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;

use crate::{Arena, ast};
use crate::types::{TypeRef, Type};
use crate::type_checking::{Expression, ExprRef, ExprToType, Item, TypedExpression};

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub(crate) struct Var(usize);

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

#[derive(Debug)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Const {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Struct,
    Undefined,
}

impl Expression<'_> {
    pub(crate) fn as_const(&self) -> Option<Const> {
        match self {
            Expression::Integer(x) => Some(Const::I32(*x as _)),
            Expression::Float(x) => Some(Const::F32(*x as _)),
            Expression::Prefix(ast::Operator::Negate, inner) => match inner.as_const()? {
                Const::I32(val) => Some(Const::I32(-val)),
                _ => None
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
            Instr::BinaryOperation(left, op, a, b) => write!(f, "{:?} = {:?} {} {:?}", left, a, match op {
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
                _ => unimplemented!()
            }, b),
            Instr::UnaryOperation(left, op, a) => write!(f, "{:?} = {}{:?}", left, match op {
                ast::Operator::Negate => "-",
                ast::Operator::Deref => "*",
                ast::Operator::Ref => "&",
                _ => unimplemented!("{:?}", op),
            }, a),
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

pub(crate) fn dump_ir(ir: &FunctionIr<'_>, f: &mut impl Write) -> io::Result<()> {
    write!(f, "fn {}(", ir.name)?;
    for (idx, it) in ir.defines.iter().enumerate().take(ir.num_args) {
        write!(f, "_{}: {:?}, ", idx, it.ty)?;
    }
    writeln!(f, ") -> {:?} {{", ir.defines[ir.num_args].ty)?;

    for (idx, it) in ir.defines.iter().enumerate().skip(ir.num_args + 1) {
        writeln!(f, "  let _{}: {:?}; // {}", idx, it.ty, &it.name.as_deref().unwrap_or(""))?;
    }

    for (idx, block) in ir.blocks.iter().enumerate() {
        writeln!(f, "  _bb{} {{", idx)?;
        for inst in &block.instrs {
            writeln!(f, "    {:?};", inst)?;
        }
        match block.terminator {
            Terminator::Jump(b) => writeln!(f, "    Jump(_bb{});", b.0)?,
            Terminator::JumpIf(v, bt, bf) => writeln!(f, "    JumpIf(_{}, _bb{}, _bb{});", v.0, bt.0, bf.0)?,
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
        Self { args: 0, vars: vec![], blocks: Default::default() }
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
        self.vars.push(VarDef { ty, name: name.map(String::from) });
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
        self.blocks.push(BlockBody { instrs: vec![], terminator: Terminator::Unreachable });
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
    expr: &'expr Expression<'expr>,
    builder: &mut IrBuilder<'tcx>,
    names: &HashMap<String, Var>,
    block: Block,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> Var {
    match expr {
        Expression::Identifier(ident) => {
            names[ident]
        }
        Expression::Integer(val) => {
            let var = builder.make_var(type_by_expr.of(expr), None);
            builder.push(block, Instr::Const(var, Const::I32(*val as i32)));
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
            builder.push(block, Instr::Cast(var, x, match (type_by_expr.of(source_expr), type_by_expr.of(expr)) {
                (Type::F32, Type::I32) => CastType::F32ToI32,
                (Type::I32, Type::F32) => CastType::I32ToF32,
                (a, b) => todo!("{:?} {:?}", a, b),
            }));
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
    let make_expr = |exprs: &'expr Arena<Expression<'expr>>,
                     type_by_expr: &mut ExprToType<'tcx>,
                     tye: TypedExpression<'expr, 'tcx>| -> ExprRef<'expr> {
        let expr = exprs.alloc(tye.expr);
        type_by_expr.insert(expr, tye.ty);
        expr
    };

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
        Item::Assignment { lhs, operator, expr } => {
            let rhs = visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
            let lhs = match &lhs {
                Expression::Identifier(ident) => local_names[ident],
                Expression::Prefix(ast::Operator::Deref, expr) => {
                    todo!("deref");
                }
                _ => unimplemented!(),
            };
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
        Item::If { condition, arm_true, arm_false } => {
            let cond_var = visit_expr(&condition, builder, &local_names, block, exprs, type_by_expr);

            let first_block_true = builder.block();
            let succ_block = builder.block();

            let mut block_true = first_block_true;
            builder.set_terminator_of(first_block_true, Terminator::Jump(succ_block));
            for item in arm_true {
                block_true = visit_item(item, arena, builder, local_names, ret, after_loop, block_true, exprs, type_by_expr);
            }

            let block_false = if let Some(items) = arm_false {
                let first_block_false = builder.block();

                let mut block_false = first_block_false;
                builder.set_terminator_of(block_false, Terminator::Jump(succ_block));
                for item in items {
                    block_false = visit_item(item, arena, builder, local_names, ret, after_loop, block_false, exprs, type_by_expr);
                }

                first_block_false
            } else {
                succ_block
            };

            builder.set_terminator_of(block, Terminator::JumpIf(cond_var, first_block_true, block_false));
            succ_block
        }
        Item::Return(expr) => {
            let var = visit_expr(&expr, builder, &local_names, block, exprs, type_by_expr);
            builder.push(block, Instr::Copy(ret.unwrap(), var));
            builder.set_terminator_of(block, Terminator::Return);
            block
        }
        Item::ForIn { name, expr, body } => {
            match type_by_expr.of(expr) {
                &Type::Array(len, item_ty) => {
                    let items_id = String::from("_items");
                    let index_id = String::from("_x");

                    let expr_ty = type_by_expr.of(expr);
                    let items = vec![
                        Item::Let {
                            name: items_id.clone(),
                            ty: expr_ty,
                            expr: Some(expr.clone()),
                        },
                        Item::Let {
                            name: index_id.clone(),
                            ty: arena.alloc(Type::I32),
                            expr: Some(make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Integer(0) })),
                        },
                        Item::Loop {
                            body: {
                                let expr = Expression::Infix(
                                    ast::Operator::Equal,
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Integer(len as i64) }),
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index_id.clone()) }),
                                );

                                let e = Expression::Index(
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: expr_ty, expr: Expression::Identifier(items_id) }),
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::U32, expr: Expression::Identifier(index_id.clone()) }),
                                );

                                let mut items = vec![
                                    Item::If {
                                        condition: make_expr(exprs, type_by_expr, TypedExpression {
                                            ty: &Type::Bool,
                                            expr,
                                        }),
                                        arm_true: vec![Item::Break],
                                        arm_false: None,
                                    },
                                    Item::Let {
                                        name: name.to_string(),
                                        ty: item_ty,
                                        expr: Some(make_expr(exprs, type_by_expr, TypedExpression {
                                            ty: item_ty,
                                            expr: e,
                                        })),
                                    },
                                ];
                                items.extend_from_slice(body);
                                let e = Expression::Infix(
                                    ast::Operator::Add,
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index_id.clone()) }),
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Integer(1) }),
                                );
                                items.push(Item::Assignment {
                                    lhs: make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index_id) }),
                                    operator: None,
                                    expr: make_expr(exprs, type_by_expr, TypedExpression {
                                        ty: &Type::I32,
                                        expr: e,
                                    }),
                                });
                                items
                            }
                        },
                    ];

                    visit_item(&Item::Block(items), arena, builder, local_names, ret, None, block, exprs, type_by_expr)
                }
                &Type::Range => {
                    let Expression::Range(start_expr, end_expr) = &expr else {
                        todo!();
                    };
                    let end_expr = end_expr.as_ref().unwrap();

                    let start = visit_expr(&start_expr, builder, local_names, block, exprs, type_by_expr);
                    let end = visit_expr(&end_expr, builder, local_names, block, exprs, type_by_expr);

                    let index = name.clone();

                    let items = vec![
                        Item::Let {
                            name: index.clone(),
                            ty: &Type::I32,
                            expr: Some(make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Integer(0) })),
                        },
                        Item::Loop {
                            body: {
                                let e = Expression::Infix(
                                    ast::Operator::Equal,
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index.clone()) }),
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Var(end) }),
                                );
                                let mut items = vec![
                                    Item::If {
                                        condition: make_expr(exprs, type_by_expr, TypedExpression {
                                            ty: &Type::Bool,
                                            expr: e,
                                        }),
                                        arm_true: vec![Item::Break],
                                        arm_false: None,
                                    },
                                ];
                                items.extend_from_slice(body);
                                let e = Expression::Infix(
                                    ast::Operator::Add,
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index.clone()) }),
                                    make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Integer(1) }),
                                );
                                items.push(Item::Assignment {
                                    lhs: make_expr(exprs, type_by_expr, TypedExpression { ty: &Type::I32, expr: Expression::Identifier(index) }),
                                    operator: None,
                                    expr: make_expr(exprs, type_by_expr, TypedExpression {
                                        ty: &Type::I32,
                                        expr: e,
                                    }),
                                });
                                items
                            }
                        },
                    ];

                    visit_item(&Item::Block(items), arena, builder, local_names, ret, None, block, exprs, type_by_expr)
                }
                other => {
                    log::error!("Unsupported {:?}", other);
                    block
                }
            }
        }
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
                        current = visit_item(other, arena, builder, local_names, ret, Some(after), current, exprs, type_by_expr);
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
                block = visit_item(item, arena, builder, local_names, ret, after_loop, block, exprs, type_by_expr);
            }
            block
        }
        Item::Function { .. } => {
            block
        }
        Item::Assert(expr) => {
            block
        }
        other => unimplemented!("{:?}", other),
    }
}

pub(crate) fn build_ir<'expr, 'tcx>(item: &Item<'expr, 'tcx>, arena: &'tcx Arena<Type<'tcx>>,
                             exprs: &'expr Arena<Expression<'expr>>,
                             type_by_expr: &mut ExprToType<'tcx>,
) -> Result<FunctionIr<'tcx>, ()> {
    let mut builder = IrBuilder::new();
    match item {
        Item::Function { name, is_extern, args, ty, body } if !is_extern => {
            let mut names = HashMap::new();
            for arg in args {
                let var = builder.make_arg(arg.ty, Some(name.as_str()));
                names.insert(arg.name.clone(), var);
            }
            let ret = builder.make_ret(ty);

            let mut block = builder.block();
            for item in body {
                block = visit_item(item, arena, &mut builder, &mut names, Some(ret), None, block, exprs, type_by_expr);
            }

            return Ok(builder.build(name.to_owned()));
        }
        _ => eprintln!("trying generate ir of item that is not a function")
    }
    Err(())
}


pub(crate) fn execute_ir(ir: &FunctionIr<'_>, args: &[Const], functions: &HashMap<String, FunctionIr<'_>>) -> Const {
    let mut curr_block = 0;
    let mut curr_inst = 0;
    let mut vars: HashMap<Var, Const> = HashMap::new();
    let mut vars_arrays: HashMap<(Var, usize), Const> = HashMap::new();

    for (idx, cnst) in args.iter().enumerate().take(ir.num_args) {
        vars.insert(Var(idx), *cnst);
    }

    loop {
        match ir.blocks[curr_block].instrs.get(curr_inst) {
            Some(instr) => {
                match instr {
                    Instr::Const(dst, val) => {
                        vars.insert(*dst, *val);
                    }
                    Instr::Copy(dst, src) => {
                        match vars.get(src).copied() {
                            Some(var) => {
                                vars.insert(*dst, var);

                                if let Const::Struct = var {
                                    // Copy all fields
                                    let mut idx = 0;
                                    while let Some(c) = vars_arrays.get(&(*src, idx)) {
                                        vars_arrays.insert((*dst, idx), c.clone());
                                        idx += 1;
                                    }
                                }
                            }
                            None => {
                                let mut to_copy = vec![];
                                for ((var, idx), _) in vars_arrays.iter() {
                                    if var == src {
                                        to_copy.push(*idx);
                                    }
                                }

                                for to in to_copy {
                                    vars_arrays.insert((*dst, to), vars_arrays[&(*src, to)]);
                                }
                            }
                        }
                    }
                    Instr::UnaryOperation(dst, op, a) => {
                        let val = match (op, vars[a]) {
                            (ast::Operator::Negate, Const::I32(v)) => Const::I32(-v),
                            (ast::Operator::Negate, Const::F32(v)) => Const::F32(-v),
                            (ast::Operator::Ref, _) => {
                                log::error!("unsupported ref op");
                                Const::Undefined
                            }
                            _ => {
                                unimplemented!("{:?} {:?}", op, a);
                            }
                        };
                        vars.insert(*dst, val);
                    }
                    Instr::BinaryOperation(dst, op, a, b) => {
                        let a = vars.get(a).copied().unwrap_or(Const::Undefined);
                        let b = vars.get(b).copied().unwrap_or(Const::Undefined);
                        let val = match (op, a, b) {
                            (ast::Operator::Add, Const::I32(a), Const::I32(b)) => Const::I32(a + b),
                            (ast::Operator::Mul, Const::I32(a), Const::I32(b)) => Const::I32(a * b),
                            (ast::Operator::Sub, Const::I32(a), Const::I32(b)) => Const::I32(a.saturating_sub(b)),
                            (ast::Operator::Div, Const::I32(a), Const::I32(b)) => Const::I32(a / b),
                            (ast::Operator::Less, Const::I32(a), Const::I32(b)) => Const::Bool(a < b),
                            (ast::Operator::Greater, Const::I32(a), Const::I32(b)) => Const::Bool(a > b),
                            (ast::Operator::Equal, Const::I32(a), Const::I32(b)) => Const::Bool(a == b),
                            (ast::Operator::NotEqual, Const::I32(a), Const::I32(b)) => Const::Bool(a != b),
                            (ast::Operator::LessEqual, Const::I32(a), Const::I32(b)) => Const::Bool(a <= b),
                            (ast::Operator::GreaterEqual, Const::I32(a), Const::I32(b)) => Const::Bool(a >= b),

                            (ast::Operator::Add, Const::U32(a), Const::U32(b)) => Const::U32(a + b),
                            (ast::Operator::Mul, Const::U32(a), Const::U32(b)) => Const::U32(a * b),
                            (ast::Operator::Sub, Const::U32(a), Const::U32(b)) => Const::U32(a.saturating_sub(b)),
                            (ast::Operator::Div, Const::U32(a), Const::U32(b)) => Const::U32(a / b),
                            (ast::Operator::Less, Const::U32(a), Const::U32(b)) => Const::Bool(a < b),
                            (ast::Operator::Greater, Const::U32(a), Const::U32(b)) => Const::Bool(a > b),
                            (ast::Operator::Equal, Const::U32(a), Const::U32(b)) => Const::Bool(a == b),
                            (ast::Operator::NotEqual, Const::U32(a), Const::U32(b)) => Const::Bool(a != b),
                            (ast::Operator::LessEqual, Const::U32(a), Const::U32(b)) => Const::Bool(a <= b),
                            (ast::Operator::GreaterEqual, Const::U32(a), Const::U32(b)) => Const::Bool(a >= b),

                            (ast::Operator::Add, Const::F32(a), Const::F32(b)) => Const::F32(a + b),
                            (ast::Operator::Mul, Const::F32(a), Const::F32(b)) => Const::F32(a * b),
                            (ast::Operator::Sub, Const::F32(a), Const::F32(b)) => Const::F32(a - b),
                            (ast::Operator::Div, Const::F32(a), Const::F32(b)) => Const::F32(a / b),
                            (ast::Operator::Less, Const::F32(a), Const::F32(b)) => Const::Bool(a < b),
                            (ast::Operator::Greater, Const::F32(a), Const::F32(b)) => Const::Bool(a > b),
                            (ast::Operator::Equal, Const::F32(a), Const::F32(b)) => Const::Bool(a == b),
                            (ast::Operator::NotEqual, Const::F32(a), Const::F32(b)) => Const::Bool(a != b),
                            (ast::Operator::LessEqual, Const::F32(a), Const::F32(b)) => Const::Bool(a <= b),
                            (ast::Operator::GreaterEqual, Const::F32(a), Const::F32(b)) => Const::Bool(a >= b),

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
                        };
                        vars.insert(*dst, val);
                    }
                    Instr::SetElement(arr, index, val) => {
                        vars_arrays.insert((*arr, *index), vars[val]);
                    }
                    Instr::GetElement(var, arr, index) => {
                        let index = match vars[index] {
                            Const::U32(v) => v as usize,
                            Const::I32(v) => v as usize,
                            other => unimplemented!("{:?}", other),
                        };
                        vars.insert(*var, vars_arrays[&(*arr, index)]);
                    }
                    Instr::Cast(target, source, mode) => {
                        let source = vars.get(source).copied().unwrap_or(Const::Undefined);
                        vars.insert(*target, match (mode, source) {
                            (_, Const::Undefined) => Const::Undefined,
                            (CastType::F32ToI32, Const::F32(val)) => Const::I32(val as i32),
                            (CastType::I32ToF32, Const::I32(val)) => Const::F32(val as f32),
                            _ => todo!(),
                        });
                    }
                    Instr::Call(target, name, args) => {
                        match name.as_str() {
                            "intrinsic.debug" => {
                                assert_eq!(args.len(), 1);
                                let value = vars[&args[0]];
                                log::debug!("Debug value: {:?}", &value);
                                vars.insert(*target, value);
                            }
                            name => {
                                let func = &functions[name];
                                let args: Vec<_> = args.iter().map(|it| vars[it]).collect();
                                let result = execute_ir(func, &args, functions);
                                vars.insert(*target, result);
                            }
                        }
                    }
                    Instr::SetField(lhs, idx, rhs) => {
                        vars_arrays.insert((*lhs, *idx), vars[rhs]);
                        vars.insert(*lhs, Const::Struct);
                    }
                    Instr::GetField(target, lhs, idx) => {
                        vars.insert(*target, vars_arrays.get(&(*lhs, *idx)).cloned().unwrap());
                    }
                }
                curr_inst += 1;
            }
            None => match ir.blocks[curr_block].terminator {
                Terminator::Jump(block) => {
                    curr_block = block.0;
                    curr_inst = 0;
                }
                Terminator::JumpIf(var, if_true, if_else) => {
                    match vars[&var] {
                        Const::Bool(v) => {
                            curr_block = if v { if_true.0 } else { if_else.0 };
                            curr_inst = 0;
                        }
                        Const::Undefined => {
                            log::error!("trying to jump to undefined");
                            return Const::Undefined;
                        }
                        other => unimplemented!("{:?}", other),
                    }
                }
                Terminator::Return => {
                    return match vars.get(&Var(ir.num_args)) {
                        Some(x) => *x,
                        None => Const::Undefined
                    };
                }
                Terminator::Unreachable => {
                    log::warn!("executing unreachable");
                    return Const::Undefined;
                }
                Terminator::Assert(var, block) => {
                    match vars[&var] {
                        Const::Bool(true) => {
                            curr_block = block.0;
                            curr_inst = 0;
                        }
                        Const::Bool(false) => {
                            log::error!("Assertion failed!!!!");
                        }
                        other => unimplemented!("{:?}", other),
                    }
                }
            }
        }
    }
}
