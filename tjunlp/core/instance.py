from typing import MutableMapping


class Instance(object):
    """
    An ``Instance`` is a sample in the dataset, a model can then decide which
    fields it wants to use as inputs as which as outputs.
    tokens, characters, sentence, label etc.

    Parameters
    ----------
    fields : ``MutableMapping[str, object]``
        The ``Field`` objects that will be used to produce data arrays for this instance.
    """

    def __init__(self, fields: MutableMapping[str, object]) -> None:
        self.fields = fields

    def __getitem__(self, key: str):
        if key in self.fields:
            return self.fields[key]
        else:
            raise KeyError("{} not found".format(key))

    def __setitem__(self, name, field):
        self.add_field(name, field)

    def __contains__(self, item):
        return item in self.fields

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def add_field(self, field_name: str, field: object) -> None:
        self.fields[field_name] = field

    def __str__(self) -> str:
        base_string = f"Instance with fields:\n"
        return " ".join(
            [base_string] + [f"\t {name}: {field} \n" for name, field in self.fields.items()]
        )

    def __repr__(self) -> str:
        return self.__str__()
