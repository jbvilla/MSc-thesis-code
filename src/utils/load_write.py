import os
import pickle
from fpdf import FPDF


def load_pickle(file_path):
    """
    Load a pickle file.
    :param file_path: path to the pickle file
    :return: the loaded object
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


def write_pickle(obj, path):
    """
    Write an object to a pickle file.
    :param obj: the object to write
    :param path: path to the pickle file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def write_result(current_best: object, result_directory: str, generation: int, name: str = "params", last: bool = False) -> None:
    """
    Write the result to a pickle file and save it to the VSC_DATA directory if it is running on the HPC.
    :param current_best: Result to save
    :param result_directory: The directory to save the result to
    :param generation: The generation
    :param name: The name of the file
    :param last: Whether this is the last result
    :return:
    """

    if last:
        result_path = os.path.join(result_directory, "params", f"{name}_last.pkl")
    else:
        result_path = os.path.join(result_directory, "params", f"{name}_gen_{generation}.pkl")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    write_pickle(current_best, result_path)


def get_result_directory(experiment_name: str, run_name: str, current_directory: str):
    """
    Returns the destination directory for the results. (HPC or local) and creates the directory if it does not exist.
    :param experiment_name: The name of the experiment
    :param run_name: The name of the run
    :param current_directory: The current directory
    :return: The destination directory
    """

    # Check if it is running on the HPC, if so, save the result to the VSC_DATA directory
    vsc_data_path = os.environ.get('VSC_DATA')
    if vsc_data_path:
        result_directory = os.path.join(vsc_data_path, "masterproef", "results", experiment_name, run_name)
    else:
        result_directory = os.path.join(current_directory, "results", run_name)

    return result_directory


def write_pdf_with_url(run_name: str, url: str, result_directory: str) -> None:
    """
    Write a pdf with the run name and the wandb url.
    :param run_name: The name of the run
    :param url: The url of the wandb run
    :param result_directory: The directory to save the pdf
    :return: None
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Run name: {run_name}", ln=True)
    pdf.cell(200, 10, txt=f"Wandb url: {url}", ln=True)

    pdf.output(os.path.join(result_directory, "wandb_url.pdf"))


def save_dict_data_to_csv(dictionary: dict, name: str, result_directory: str) -> None:
    """
    Save dictionary data to a csv file.
    :param dictionary: The dictionary to save
    :param name: The name of the file
    :param result_directory: The directory to save the file
    :return: None
    """

    file_name = os.path.join(result_directory, f"{name}.csv")
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(",".join(dictionary.keys()) + "\n")

    with open(file_name, "a") as f:
        f.write(",".join(map(str, dictionary.values())) + "\n")

