from run import Runner
import sys


def evaluate(config_name, saved_suffix, gpu_id):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    print("=================================")
    runner.evaluate(
        model,
        examples_test,
        stored_info,
        0,
        official=True,
        conll_path=runner.config["conll_test_path"],
    )  # Eval test


if __name__ == "__main__":
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluate(config_name, saved_suffix, gpu_id)
