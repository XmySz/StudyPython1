def nearest_value(values: set, one: int) -> int:
    n = [abs(one-i) for i in values]
    m = min(n)
    num = list(values)[n.index(m)]
    n.reverse()
    m1 = min(n)
    l = list(values)
    l.reverse()
    num1 = l[n.index(m1)]
    return min(num, num1)


if __name__ == "__main__":
    print("Example:")
    print(nearest_value({4, 7, 10, 11, 12, 17}, 9))

    # These "asserts" are used for self-checking and not for an auto-testing
    assert nearest_value({4, 7, 10, 11, 12, 17}, 9) == 10
    assert nearest_value({4, 7, 10, 11, 12, 17}, 8) == 7
    assert nearest_value({4, 8, 10, 11, 12, 17}, 9) == 8
    assert nearest_value({4, 9, 10, 11, 12, 17}, 9) == 9
    assert nearest_value({4, 7, 10, 11, 12, 17}, 0) == 4
    assert nearest_value({4, 7, 10, 11, 12, 17}, 100) == 17
    assert nearest_value({5, 10, 8, 12, 89, 100}, 7) == 8
    assert nearest_value({5}, 5) == 5
    assert nearest_value({5}, 7) == 5
    assert nearest_value({-2, 0}, -1) == -2
    print("Coding complete? Click 'Check' to earn cool rewards!")
