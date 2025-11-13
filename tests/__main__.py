import pathlib
import unittest


def main():
    tests_path = pathlib.Path(__file__).parent
    suite = unittest.defaultTestLoader.discover(start_dir=str(tests_path))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

