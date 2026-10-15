import traceback

def print_results(result, title):
    print("\033[93mResult of test", title, ":", result, "\033[0m")


def test(func, title, verbose = False, verbose_func = print_results):
    try:
        result = func()
        if verbose:
            print("\033[92mTest \"" + title + "\" Successful.\033[0m")
            if verbose_func:
                verbose_func(result, title)
        return result
    except Exception as e:
        print("\033[91mTest \"" + title + "\" Failed:", e, "\033[0m")
        if verbose:
            traceback.print_exception(type(e), e, e.__traceback__)

if __name__ == "__main__":
    # Define a multiline function
    def multiline_func():
        x = 1
        y = 2
        return x / y

    # Use the new functions
    test(lambda: 1/0, "Division by Zero Test", verbose=True)
    test(lambda: 1/1, "Division by One Test", verbose=True)

    # Use the multiline function
    test(multiline_func, "Multiline Function Test", verbose=True)

    # Use inline multiline function
    test(lambda:
            (x := 1,
            y := 2,
            x / y),
        "Inline Multiline Function Test",
        verbose=True)
    
    