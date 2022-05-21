use crate::ty::{Ty, TyS};
use crate::ast::{Operator, Item, Expr, TyExpr};
use std::{fmt, io};
use std::io::Write;
use std::collections::HashMap;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub(crate) struct Var(usize);

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "_{}", self.0)
    }
}

#[derive(Debug)]
#[derive(Copy, Clone)]
pub(crate) enum Const {
    U32(u32),
    F32(f32),
    Bool(bool),
}

enum Instr {
    Const(Var, Const),
    Copy(Var, Var),
    UnaryOperation(Var, Operator, Var),
    BinaryOperation(Var, Operator, Var, Var),
    SetElement(Var, usize, Var),
    GetElement(Var, Var, Var),
    Debug(Vec<Var>),
}

impl fmt::Debug for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Instr::Const(var, val) => write!(f, "{:?} = {:?}", var, val),
            Instr::Copy(lhs, rhs) => write!(f, "{:?} = {:?}", lhs, rhs),
            Instr::BinaryOperation(left, op, a, b) => write!(f, "{:?} = {:?} {} {:?}", left, a, match op {
                Operator::Add => "+",
                Operator::Sub => "-",
                Operator::Mul => "*",
                Operator::Div => "/",
                Operator::Equal => "==",
                Operator::NotEqual => "!=",
                Operator::Less => "<",
                Operator::LessEqual => "<=",
                Operator::Greater => ">",
                Operator::GreaterEqual => ">=",
                _ => unimplemented!()
            }, b),
            Instr::UnaryOperation(left, op, a) => write!(f, "{:?} = {}{:?}", left, match op {
                Operator::Negate => "-",
                Operator::Deref => "*",
                _ => unimplemented!()
            }, a),
            Instr::SetElement(arr, index, val) => {
                write!(f, "{:?}[{}] = {:?}", arr, index, val)
            }
            Instr::GetElement(var, arr, index) => {
                write!(f, "{:?} = {:?}[{:?}]", var, arr, index)
            }
            Instr::Debug(args) => {
                Ok(())
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
}

#[derive(Debug)]
struct VarDef<'tcx> {
    name: Option<String>,
    ty: Ty<'tcx>,
}

#[derive(Debug)]
struct BlockBody {
    instrs: Vec<Instr>,
    terminator: Terminator,
}

#[derive(Debug)]
pub(crate) struct Mir<'tcx> {
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

pub(crate) fn dump_mir(mir: &Mir<'_>, f: &mut impl Write) -> io::Result<()> {
    write!(f, "fn {}(", mir.name)?;
    for (idx, it) in mir.defines.iter().enumerate().take(mir.num_args) {
        write!(f, "_{}: {:?}, ", idx, it.ty)?;
    }
    writeln!(f, ") -> {:?} {{", mir.defines[mir.num_args].ty)?;

    for (idx, it) in mir.defines.iter().enumerate().skip(mir.num_args + 1) {
        writeln!(f, "  let _{}: {:?}; // {}", idx, it.ty, &it.name.as_deref().unwrap_or(""))?;
    }

    for (idx, block) in mir.blocks.iter().enumerate() {
        writeln!(f, "  _bb{} {{", idx)?;
        for inst in &block.instrs {
            writeln!(f, "    {:?};", inst)?;
        }
        match block.terminator {
            Terminator::Jump(b) => writeln!(f, "    Jump(_bb{});", b.0)?,
            Terminator::JumpIf(v, bt, bf) => writeln!(f, "    JumpIf(_{}, _bb{}, _bb{});", v.0, bt.0, bf.0)?,
            Terminator::Return => writeln!(f, "    Return;")?,
            Terminator::Unreachable => writeln!(f, "    Unreachable;")?,
        }
        writeln!(f, "  }}")?;
    }

    writeln!(f, "}}")?;

    Ok(())
}

struct MirBuilder<'tcx> {
    args: usize,
    vars: Vec<VarDef<'tcx>>,
    blocks: Vec<BlockBody>,
}

impl<'tcx> MirBuilder<'tcx> {
    fn new() -> Self {
        Self { args: 0, vars: vec![], blocks: Default::default() }
    }

    fn make_arg(&mut self, ty: Ty<'tcx>, name: Option<&str>) -> Var {
        assert_eq!(self.args, self.vars.len());
        self.args += 1;
        self.make_var(ty, name)
    }

    fn make_ret(&mut self, ty: Ty<'tcx>) -> Var {
        assert_eq!(self.args, self.vars.len());
        self.make_var(ty, None)
    }

    fn make_var(&mut self, ty: Ty<'tcx>, name: Option<&str>) -> Var {
        let var = Var(self.vars.len());
        self.vars.push(VarDef { ty, name: name.map(String::from) });
        var
    }

    fn build(self, name: String) -> Mir<'tcx> {
        Mir {
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

fn visit_expr<'tcx>(
    expr: &TyExpr<'tcx>,
    builder: &mut MirBuilder<'tcx>,
    names: &HashMap<String, Var>,
    block: Block,
) -> Var {
    match &expr.expr {
        Expr::Identifier(ident) => {
            names[ident]
        }
        Expr::Integer(val) => {
            let var = builder.make_var(expr.ty, None);
            builder.push(block, Instr::Const(var, Const::U32(*val as u32)));
            var
        }
        Expr::Float(val) => {
            let var = builder.make_var(expr.ty, None);
            builder.push(block, Instr::Const(var, Const::F32(*val as f32)));
            var
        }
        Expr::Bool(val) => {
            let var = builder.make_var(expr.ty, None);
            builder.push(block, Instr::Const(var, Const::Bool(*val)));
            var
        }
        Expr::Prefix(op, rhs) => {
            let var = builder.make_var(expr.ty, None);
            let operand = visit_expr(rhs, builder, &names, block);
            builder.push(block, Instr::UnaryOperation(var, *op, operand));
            var
        }
        Expr::Infix(op, lhs, rhs) => {
            let var = builder.make_var(expr.ty, None);
            let a = visit_expr(lhs, builder, &names, block);
            let b = visit_expr(rhs, builder, &names, block);
            builder.push(block, Instr::BinaryOperation(var, *op, a, b));
            var
        }
        Expr::Array(elements) => {
            let var = builder.make_var(expr.ty, None);
            for (index, item) in elements.iter().enumerate() {
                let expr = visit_expr(item, builder, names, block);
                builder.push(block, Instr::SetElement(var, index, expr));
            }
            var
        }
        Expr::Index(slice, index) => {
            let ty = match slice.ty {
                TyS::Array(_, ty) => ty,
                TyS::Slice(_) => unimplemented!(),
                _ => unreachable!(),
            };

            let slice_var = match &slice.expr {
                Expr::Identifier(ident) => names[ident],
                _ => unimplemented!(),
            };

            let element_var = builder.make_var(ty, None);
            let index_var = visit_expr(index, builder, names, block);
            builder.push(block, Instr::GetElement(element_var, slice_var, index_var));
            element_var
        }
        Expr::Call(func, args) => {
            let mut params = Vec::new();
            for arg in args {
                params.push(visit_expr(arg, builder, names, block));
            }

            builder.push(block, Instr::Debug(params));
            builder.make_var(expr.ty, Some("Return of"))
        }
        Expr::Place(_, _) | Expr::Tuple(_) | Expr::Range(_, _) => unimplemented!(),
        Expr::Var(var) => var.clone(),
    }
}

fn visit_item<'tcx>(
    item: &Item<'tcx>,
    builder: &mut MirBuilder<'tcx>,
    local_names: &mut HashMap<String, Var>,
    ret: Option<Var>,
    after_loop: Option<Block>,
    block: &mut Block,
) {
    match item {
        Item::Let { name, ty, expr } => {
            let var = builder.make_var(ty.unwrap(), Some(name.as_str()));
            local_names.insert(name.clone(), var);

            if let Some(expr) = expr {
                let expr = visit_expr(expr, builder, &local_names, *block);
                builder.push(*block, Instr::Copy(var, expr));
            }
        }
        Item::Assignment { lhs, operator, expr } => {
            let rhs = visit_expr(expr, builder, &local_names, *block);
            let lhs = match &lhs.expr {
                Expr::Identifier(ident) => local_names[ident],
                _ => unimplemented!(),
            };
            if let Some(op) = operator {
                builder.push(*block, Instr::BinaryOperation(lhs, *op, lhs, rhs));
            } else {
                builder.push(*block, Instr::Copy(lhs, rhs));
            }
        }
        Item::Expr { expr } => {
            visit_expr(expr, builder, &local_names, *block);
        }
        Item::If { condition, arm_true, arm_false } => {
            let cond_var = visit_expr(condition, builder, &local_names, *block);
            let mut block_true = builder.block();
            for item in arm_true {
                visit_item(item, builder, local_names, ret, after_loop, &mut block_true);
            }

            let block_false = if let Some(arm_false) = arm_false {
                let mut block_false = builder.block();
                for item in arm_false {
                    visit_item(item, builder, local_names, ret, after_loop, &mut block_false);
                }
                Some(block_false)
            } else {
                None
            };
            let next_block = builder.block();

            let block_false = if let Some(b) = block_false { b } else { next_block };
            builder.set_terminator_of(*block, Terminator::JumpIf(cond_var, block_true, block_false));
            *block = next_block;
        }
        Item::Return(expr) => {
            let var = visit_expr(expr, builder, &local_names, *block);
            builder.push(*block, Instr::Copy(ret.unwrap(), var));
            builder.set_terminator_of(*block, Terminator::Return);
        }
        Item::ForIn { name, expr, body } => {
            match expr.ty {
                &TyS::Array(len, item_ty) => {
                    let items_id = String::from("_items");
                    let index_id = String::from("_x");

                    let items = vec![
                        Item::Let {
                            name: items_id.clone(),
                            ty: Some(expr.ty),
                            expr: Some(expr.clone()),
                        },
                        Item::Let {
                            name: index_id.clone(),
                            ty: Some(&TyS::I32),
                            expr: Some(TyExpr { ty: &TyS::I32, expr: Expr::Integer(0) }),
                        },
                        Item::Loop {
                            body: {
                                let mut items = vec![
                                    Item::If {
                                        condition: TyExpr {
                                            ty: &TyS::Bool,
                                            expr: Expr::Infix(
                                                Operator::Equal,
                                                Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Integer(len as i64) }),
                                                Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index_id.clone()) }),
                                            ),
                                        },
                                        arm_true: vec![Item::Break],
                                        arm_false: None,
                                    },
                                    Item::Let {
                                        name: name.to_string(),
                                        ty: Some(item_ty),
                                        expr: Some(TyExpr {
                                            ty: item_ty,
                                            expr: Expr::Index(
                                                Box::new(TyExpr { ty: expr.ty, expr: Expr::Identifier(items_id) }),
                                                Box::new(TyExpr { ty: &TyS::U32, expr: Expr::Identifier(index_id.clone()) }),
                                            ),
                                        }),
                                    },
                                ];
                                items.extend_from_slice(body);
                                items.push(Item::Assignment {
                                    lhs: TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index_id.clone()) },
                                    operator: None,
                                    expr: TyExpr {
                                        ty: &TyS::I32,
                                        expr: Expr::Infix(
                                            Operator::Add,
                                            Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index_id) }),
                                            Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Integer(1) }),
                                        ),
                                    },
                                });
                                items
                            }
                        },
                    ];

                    visit_item(&Item::Block(items), builder, local_names, ret, None, block);
                }
                &TyS::Range => {
                    let Expr::Range(start_expr, end_expr) = &expr.expr else {
                        todo!();
                    };
                    let end_expr = end_expr.as_ref().unwrap();

                    let start = visit_expr(start_expr, builder, local_names, *block);
                    let end = visit_expr(end_expr, builder, local_names, *block);

                    let index = name.clone();

                    let items = vec![
                        Item::Let {
                            name: index.clone(),
                            ty: Some(&TyS::I32),
                            expr: Some(TyExpr { ty: &TyS::I32, expr: Expr::Integer(0) }),
                        },

                        Item::Loop {
                            body: {
                                let mut items = vec![
                                    Item::If {
                                        condition: TyExpr {
                                            ty: &TyS::Bool,
                                            expr: Expr::Infix(
                                                Operator::Equal,
                                                Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index.clone()) }),
                                                Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Var(end) }),
                                            ),
                                        },
                                        arm_true: vec![Item::Break],
                                        arm_false: None,
                                    },
                                ];
                                items.extend_from_slice(body);
                                items.push(Item::Assignment {
                                    lhs: TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index.clone()) },
                                    operator: None,
                                    expr: TyExpr {
                                        ty: &TyS::I32,
                                        expr: Expr::Infix(
                                            Operator::Add,
                                            Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Identifier(index) }),
                                            Box::new(TyExpr { ty: &TyS::I32, expr: Expr::Integer(1) }),
                                        ),
                                    },
                                });
                                items
                            }
                        },
                    ];

                    visit_item(&Item::Block(items), builder, local_names, ret, None, block);
                }
                other => unimplemented!("{:?}", other),
            };
        }
        Item::Break => {
            builder.set_terminator_of(*block, Terminator::Jump(after_loop.unwrap()));
        }
        Item::Loop { body } => {
            let entry = builder.block();
            let mut current = entry;

            let after = builder.block();

            builder.set_terminator_of(*block, Terminator::Jump(entry));
            for item in body {
                match item {
                    Item::Break => {}
                    Item::Yield(_) => {}
                    Item::Return(_) => {}
                    other => {
                        visit_item(other, builder, local_names, ret, Some(after), &mut current);
                        builder.set_terminator_of(current, Terminator::Jump(entry));
                    }
                }
            }
            builder.set_terminator_of(after, Terminator::Return);
            *block = after;
        }
        Item::Block(body) => {
            // FIXME: build a new block?
            for item in body {
                visit_item(item, builder, local_names, ret, after_loop, block);
            }

        }
        other => unimplemented!("{:?}", other),
    }
}

pub(crate) fn build_mir<'tcx>(item: &Item<'tcx>) -> Result<Mir<'tcx>, ()> {
    let mut builder = MirBuilder::new();
    match item {
        Item::Function { name, is_extern, args, ty, body } if !is_extern => {
            let mut names = HashMap::new();
            for arg in args {
                let var = builder.make_arg(arg.ty, Some(name.as_str()));
                names.insert(arg.name.clone(), var);
            }
            let ret = builder.make_ret(ty);

            let mut x = builder.block();
            for item in body {
                visit_item(item, &mut builder, &mut names, Some(ret), None, &mut x);
            }

            return Ok(builder.build(name.to_owned()));
        }
        _ => eprintln!("trying generate mir of item that is not a function")
    }
    Err(())
}


pub(crate) fn execute_mir(mir: &Mir<'_>, args: &[Const]) -> Const {
    let mut curr_block = 0;
    let mut curr_inst = 0;
    let mut vars: HashMap<Var, Const> = HashMap::new();
    let mut vars_arrays: HashMap<(Var, usize), Const> = HashMap::new();

    for (idx, cnst) in args.iter().enumerate().take(mir.num_args) {
        vars.insert(Var(idx), *cnst);
    }

    loop {
        match mir.blocks[curr_block].instrs.get(curr_inst) {
            Some(instr) => {
                match instr {
                    Instr::Const(dst, val) => {
                        vars.insert(*dst, *val);
                    }
                    Instr::Copy(dst, src) => {
                        match vars.get(src).copied() {
                            Some(var) => {
                                vars.insert(*dst, var);
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
                            (Operator::Negate, Const::U32(v)) => Const::U32(v),
                            _ => unimplemented!(),
                        };
                        vars.insert(*dst, val);
                    }
                    Instr::BinaryOperation(dst, op, a, b) => {
                        let val = match (op, vars[a], vars[b]) {
                            (Operator::Add, Const::U32(a), Const::U32(b)) => Const::U32(a + b),
                            (Operator::Mul, Const::U32(a), Const::U32(b)) => Const::U32(a * b),
                            (Operator::Sub, Const::U32(a), Const::U32(b)) => Const::U32(a.saturating_sub(b)),
                            (Operator::Div, Const::U32(a), Const::U32(b)) => Const::U32(a / b),
                            (Operator::Less, Const::U32(a), Const::U32(b)) => Const::Bool(a < b),
                            (Operator::Greater, Const::U32(a), Const::U32(b)) => Const::Bool(a > b),
                            (Operator::Equal, Const::U32(a), Const::U32(b)) => Const::Bool(a == b),
                            _ => unimplemented!(),
                        };
                        vars.insert(*dst, val);
                    }
                    Instr::SetElement(arr, index, val) => {
                        vars_arrays.insert((*arr, *index), vars[val]);
                    }
                    Instr::GetElement(var, arr, index) => {
                        let index = match vars[index] {
                            Const::U32(v) => v as usize,
                            other => unimplemented!("{:?}", other),
                        };
                        vars.insert(*var, vars_arrays[&(*arr, index)]);
                    }
                    Instr::Debug(args) => {
                        for (idx, arg) in args.iter().enumerate() {
                            println!("#{} = {:?}", idx, vars[arg]);
                        }
                    }
                }
                curr_inst += 1;
            }
            None => match mir.blocks[curr_block].terminator {
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
                        other => unimplemented!("{:?}", other),
                    }
                }
                Terminator::Return => {
                    return vars[&Var(mir.num_args)];
                }
                Terminator::Unreachable => unreachable!(),
            }
        }
    }
}