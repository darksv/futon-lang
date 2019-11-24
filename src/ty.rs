#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) enum TyS<'t> {
    Bool,
    U32,
    I32,
    F32,
    Array(usize, Ty<'t>),
    Slice(Ty<'t>),
    Unit,
    Tuple(Vec<Ty<'t>>),
    Function(Vec<Ty<'t>>, Ty<'t>),
    Pointer(Ty<'t>),
    Range,
    Other(String),
}

pub(crate) type Ty<'tcx> = &'tcx TyS<'tcx>;
