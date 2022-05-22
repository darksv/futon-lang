#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) enum Type<'tcx> {
    Bool,
    U32,
    I32,
    F32,
    Array(usize, TypeRef<'tcx>),
    Slice(TypeRef<'tcx>),
    Unit,
    Tuple(Vec<TypeRef<'tcx>>),
    Function(Vec<TypeRef<'tcx>>, TypeRef<'tcx>),
    Pointer(TypeRef<'tcx>),
    Range,
    Other(String),
    Unknown,
    Error,
    Any,
}

pub(crate) type TypeRef<'tcx> = &'tcx Type<'tcx>;
