use crate::ty::{Ty, TyS};
use crate::ast::{Operator, Item, Expression};
use std::{fmt, io};
use std::io::Write;
use std::collections::HashMap;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Var(usize);

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
        }
        writeln!(f, "  }}")?;
    }

    writeln!(f, "}}")?;

    Ok(())
}

struct MirBuilder<'tcx> {
    args: usize,
    defines: Vec<VarDef<'tcx>>,
    blocks: Vec<BlockBody>,
    blocks_stack: Vec<Block>,
}

impl<'tcx> MirBuilder<'tcx> {
    fn new() -> Self {
        Self { args: 0, defines: vec![], blocks: vec![], blocks_stack: vec![] }
    }

    fn make_arg(&mut self, ty: Ty<'tcx>, name: Option<&str>) -> Var {
        assert_eq!(self.args, self.defines.len());
        self.args += 1;
        self.make_var(ty, name)
    }

    fn make_ret(&mut self, ty: Ty<'tcx>) -> Var {
        assert_eq!(self.args, self.defines.len());
        self.make_var(ty, None)
    }

    fn make_var(&mut self, ty: Ty<'tcx>, name: Option<&str>) -> Var {
        let var = Var(self.defines.len());
        self.defines.push(VarDef { ty, name: name.map(String::from) });
        var
    }

    fn build(self, name: String) -> Mir<'tcx> {
        Mir {
            name,
            num_args: self.args,
            defines: self.defines,
            blocks: self.blocks,
        }
    }

    fn begin_block(&mut self) -> Block {
        let block = Block(self.blocks.len());
        self.blocks_stack.push(block);
        self.blocks.push(BlockBody { instrs: vec![], terminator: Terminator::Return });
        block
    }

    fn end_block(&mut self) {
        self.blocks_stack.pop();
    }

    fn set_terminator(&mut self, term: Terminator) {
        self.blocks[self.blocks_stack.last().unwrap().0].terminator = term;
    }

    fn push(&mut self, inst: Instr) {
        self.blocks[self.blocks_stack.last().unwrap().0].instrs.push(inst);
    }
}


fn visit_expr(expr: &Expression, builder: &mut MirBuilder, stack: &mut Vec<Var>, names: &HashMap<String, Var>) {
    match expr {
        Expression::Identifier(id) => {
            stack.push(names[id]);
        }
        Expression::Integer(val) => {
            let var = builder.make_var(&TyS::U32, None);
            stack.push(var);
            builder.push(Instr::Const(var, Const::U32(*val as u32)));
        }
        Expression::Float(val) => {
            let var = builder.make_var(&TyS::F32, None);
            stack.push(var);
            builder.push(Instr::Const(var, Const::F32(*val as f32)));
        }
        Expression::Bool(val) => {
            let var = builder.make_var(&TyS::Bool, None);
            stack.push(var);
            builder.push(Instr::Const(var, Const::Bool(*val)));
        }
        Expression::Prefix(op, expr) => {
            let var = builder.make_var(&TyS::Unknown, None);
            builder.push(Instr::UnaryOperation(var, *op, stack.pop().unwrap()));
            stack.push(var);
        }
        Expression::Infix(op, lhs, rhs) => {
            let var = builder.make_var(&TyS::Unknown, None);
            visit_expr(lhs, builder, stack, &names);
            visit_expr(rhs, builder, stack, &names);

            let a = stack.pop().unwrap();
            let b = stack.pop().unwrap();
            builder.push(Instr::BinaryOperation(var, *op, b, a));
            stack.push(var);
        }
        other => {
            unimplemented!("{:?}", other);
        }
    }
}

fn visit_item<'tcx>(
    item: &Item<'tcx>,
    builder: &mut MirBuilder<'tcx>,
    stack: &mut Vec<Var>,
    names: &mut HashMap<String, Var>,
    ret: Option<Var>,
) {
    match item {
        Item::Let { name, ty, expr } => {
            let var = builder.make_var(ty.unwrap(), Some(name.as_str()));
            names.insert(name.clone(), var);

            if let Some(expr) = expr {
                visit_expr(expr, builder, stack, &names);
            }
            builder.push(Instr::Copy(var, stack.last().copied().unwrap()));
        }
        Item::Assignment { lhs, operator, expr } => {
            visit_expr(lhs, builder, stack, &names);
            visit_expr(expr, builder, stack, &names);
        }
        Item::Expr { expr } => {
            visit_expr(expr, builder, stack, &names);
        }
        Item::If { condition, arm_true, arm_false } => {
            visit_expr(condition, builder, stack, &names);
            let var = *stack.last().unwrap();
            let block_true = builder.begin_block();
            for item in arm_true {
                visit_item(item, builder, stack, names, ret);
            }
            builder.end_block();
            let block_false = builder.begin_block();
            if let Some(arm_false) = arm_false {
                for item in arm_false {
                    visit_item(item, builder, stack, names, ret);
                }
            }
            builder.end_block();
            builder.set_terminator(Terminator::JumpIf(var, block_true, block_false));
        }
        Item::Return(expr) => {
            visit_expr(expr, builder, stack, &names);
            builder.push(Instr::Copy(ret.unwrap(), stack.pop().unwrap()));
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
            let r = builder.make_ret(ty);

            let mut stack = vec![];
            builder.begin_block();
            for item in body {
                visit_item(item, &mut builder, &mut stack, &mut names, Some(r));
            }
            builder.end_block();

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
                        vars.insert(*dst, *vars.get(src).unwrap());
                    }
                    Instr::UnaryOperation(dst, op, a) => {
                        let val = match (op, vars[a]) {
                            (Operator::Negate, Const::U32(v)) => Const::U32(v),
                            _ => unimplemented!(),
                        };

                        dbg!(dst, op, a, val);
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
                            _ => unimplemented!(),
                        };

                        dbg!(dst, op, a, b, val);


                        vars.insert(*dst, val);
                    }
                }
                curr_inst += 1;
            }
            None => match mir.blocks[curr_block].terminator {
                Terminator::Jump(block) => {
                    curr_block = block.0;
                }
                Terminator::JumpIf(var, if_true, if_else) => {
                    match vars[&var] {
                        Const::Bool(v) => {
                            curr_block = if v { if_true.0 } else { if_else.0 };
                            curr_inst = 0;
                        }
                        _ => unimplemented!(),
                    }
                }
                Terminator::Return => {
                    return vars[&Var(mir.num_args)];
                }
            }
        }
    }
}