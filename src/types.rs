#[derive(Debug, Clone, Eq, PartialEq, Hash, Default)]
pub(crate) enum Type<'tcx> {
    Bool,
    Integer,
    Float,
    U32,
    I32,
    F32,
    Array(usize, TypeRef<'tcx>),
    Slice(TypeRef<'tcx>),
    Unit,
    Tuple(Vec<TypeRef<'tcx>>),
    Struct {
        fields: Vec<(String, TypeRef<'tcx>)>,
    },
    Function(Vec<TypeRef<'tcx>>, TypeRef<'tcx>),
    Pointer(TypeRef<'tcx>),
    Range,
    #[default]
    Unknown,
    Error,
    Any,
    Placeholder(usize),
}

impl<'tcx> Type<'tcx> {
    pub(crate) fn is_abstract(&self) -> bool {
        match self {
            Type::Integer | Type::Float => true,
            _ => false,
        }
    }
}

pub(crate) type TypeRef<'tcx> = &'tcx Type<'tcx>;
