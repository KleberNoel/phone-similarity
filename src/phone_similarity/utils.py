def print_feature_vector(columns):
    """TODO FIXME"""
    print()
    new_columns = [feat for feat in columns]
    for char_idx in range(max([len(n) for n in new_columns])):
        print("               ", end="")
        for word in new_columns:
            if char_idx < len(word):
                print(word[char_idx], end="  ")
            else:
                print(" ", end="  ")
        print()
    print()


def print_syl_verbose(
    syl, cons_columns, vowel_columns, print_zeros: bool = True
) -> None:
    """TODO FIXME"""
    print(" " * 10, "=== ONSET ===")
    for i in range(len(syl["onset"])):
        if print_zeros:
            print("            ", syl["onset"][i], cons_columns[i])
        elif syl["onset"][i] != 0:
            print("            ", syl["onset"][i], cons_columns[i])

    print(" " * 10, "=== NUCLEUS ===")
    for i in range(len(syl["nucleus"])):
        if print_zeros:
            print("            ", syl["nucleus"][i], vowel_columns[i])
        elif syl["nucleus"][i] != 0:
            print("            ", syl["nucleus"][i], vowel_columns[i])

    print(" " * 10, "=== CODA ===")
    for i in range(len(syl["coda"])):
        if print_zeros:
            print("            ", syl["coda"][i], cons_columns[i])
        elif syl["coda"][i] != 0:
            print("            ", syl["coda"][i], cons_columns[i])
