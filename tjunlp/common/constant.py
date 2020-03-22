"""
Common constants used in tjunlp.
"""

# class _Const:
#     class ConstError(TypeError):
#         pass
#
#     class ConstCaseError(ConstError):
#         pass
#
#     def __setattr__(self, name, value):
#         if name in self.__dict__:
#             raise self.ConstError("Can't change const(%s) value" % name)
#         if not name.isupper():
#             raise self.ConstCaseError("Const name %s is not all uppercase" % name)
#
#         self.__dict__[name] = value
#
#     def __getitem__(self, key):
#         if key in self.__dict__:
#             return self.__dict__[key]
#         else:
#             raise self.ConstError("Can't return const %s, No Existing Key!" % key)
#
#     def __delattr__(self, name):
#         if name in self.__dict__:
#             raise self.ConstError("Can't unbind const(%s)" % name)
#         raise NameError(name)
#
#
# const = _Const()

# dataset
KEY_TRAIN, KEY_DEV, KEY_TEST = "train", "dev", "test"
PRETRAIN_POSTFIX = "_PRETRAINED"

# vocabulary
DEFAULT_FIELD = "tokens"
