def ubyte_to_uint(byte: bytes) -> int:
    """
    Convert a byte to an unsigned integer.
    :param byte: The byte to convert.
    :return: The unsigned integer value.
    """
    return int.from_bytes(byte, byteorder='big', signed=False)