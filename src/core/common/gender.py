from enum import IntEnum


class Gender(IntEnum):
  MALE = 0
  FEMALE = 1

  def __str__(self) -> str:
    if self == Gender.MALE:
      return "M"
    if self == Gender.FEMALE:
      return "F"
    raise Exception()
