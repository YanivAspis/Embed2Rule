from os import path
import subprocess

class ILASPExample:
    def __init__(self, is_pos, ex_id, weight, inclusion_set, exclusion_set, context):
        self._is_pos = is_pos
        self._ex_id = ex_id
        self._weight = weight
        self._inclusion_set = inclusion_set
        self._exclusion_set = exclusion_set
        self._context = context

    def __str__(self):
        inclusion_set_str = "{\n" + ",\n".join(["\t" + atom for atom in self._inclusion_set]) + "\n\t}"
        exclusion_set_str = "{\n" + ",\n".join(["\t" + atom for atom in self._exclusion_set]) + "\n\t}"
        context_str = "{\n" + "\n".join(["\t" + rule for rule in self._context]) + "\n}"
        example_str = f"#{'pos' if self._is_pos else 'neg'}(eg{self._ex_id}@{self._weight}, " + inclusion_set_str + ", " + exclusion_set_str + ", " + context_str + ")."
        return example_str

class ILASPTask:
    def __init__(self, config, background_file, mode_bias_file, examples):
        self._config = config
        with open(background_file, "r") as f:
            self._background_knowledge = [line.strip() for line in f.readlines() if line.strip() != ""]
        with open(mode_bias_file, "r") as f:
            self._mode_bias = [line.strip() for line in f.readlines() if line.strip() != ""]
        self._examples = examples
        self._ilasp_command = self._setup_ilasp_command(config)

    def _setup_ilasp_command(self, config):
        if "ilasp_path" in config["symbolic_learner"]:
            ilasp_path = config["symbolic_learner"]["ilasp_path"]
        else:
            ilasp_path = "ILASP"
        ilasp_command = [ilasp_path, "-v=4"]
        if "ilasp_params" in config["symbolic_learner"]:
            ilasp_command += config["symbolic_learner"]["ilasp_params"]
        return ilasp_command

    def _parse_result(self, result_str):
        result_lines = [result_line.strip() for result_line in result_str.split("\n") if not result_line.strip().startswith('%')]
        return [line for line in result_lines if line != ""]

    def __str__(self):
        task_str = "\n".join(self._background_knowledge)
        task_str += "\n\n"
        task_str += "\n".join([str(ex) for ex in self._examples])
        task_str += "\n\n"
        task_str += "\n".join(self._mode_bias)
        return task_str

    def __call__(self, results_dir):
        with open(path.join(results_dir, "ilasp_task.las"), "w", newline="\n") as f:
            f.write(str(self))
        run_res = subprocess.run(
            self._ilasp_command + [path.join(results_dir, "ilasp_task.las")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self._config["symbolic_learner"]["timeout"]
        )
        if run_res.stderr:
            print(run_res.stderr.decode("utf-8"))
            return ""
        return self._parse_result(run_res.stdout.decode("utf-8"))
