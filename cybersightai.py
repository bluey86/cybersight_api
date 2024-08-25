import json
import os
import logging
import requests
import time
from enum import Enum
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Any
from cybersight_api.analysis_detail import AnalysisResults


class Tags(str, Enum):
    RIGHT_EYE_MACULA = "right-eye,macula-centered"
    LEFT_EYE_MACULA = "left-eye,macula-centered"
    RIGHT_EYE_DISC = "right-eye,disc-centred"
    LEFT_EYE_DISC = "left-eye,disc-centred"


class Columns(str, Enum):
    RIGHT_EYE_DISC_CENTERED = "Right Eye Disc Centered"
    RIGHT_EYE_MACULA_CENTERED = "Right Eye Macula Centered"
    LEFT_EYE_DISC_CENTERED = "Left Eye Disc Centered"
    LEFT_EYE_MACULA_CENTERED = "Left Eye Macula Centered"

    SHEEN = "Sheen \n(Prominent / Not Prominent) "

    PARTICIPANT_ID = "Participants "
    ID_CDIC = "ID_CDiC"
    ID_DDC = "ID_DDC"
    ID_LFAC = "ID_LFAC"
    SEX = "Sex"
    AGE = "Age_Yrs"

    @classmethod
    def list_columns(cls):
        columns = [member.value for column, member in cls.__members__.items()]
        return columns


class ColumnsAI(str, Enum):

    MACULOPATHY = "maculopathy"
    MACULOPATHY_LABEL = "maculopathy_label"

    AI_DR_GRADE = "icdr_value"
    AI_DR_GRADE_LABEL = "icdr_label"

    AI_REFERABLE_DR_LABEL = "referable_dr_label"

    # AI_REFERABLE_GLAUCOMA = "referable_glaucoma_score"
    AI_REFERABLE_GLAUCOMA_LABEL = "referable_glaucoma_label"

    AI_VCDR = "vcdr"

    AI_DISC_ANOMALY_SCORE = "disc_anomaly"
    AI_DISC_ANOMALY_LABEL = "disc_anomaly_label"

    AI_GRADABILITY_OVERALL = "fundus_overall_gradability"
    AI_GRADABILITY_DISC = "fundus_disc_gradability"
    AI_GRADABILITY_MACULA = "fundus_macula_gradability"
    AI_GRADABILITY_FUNDUS = "fundus_verification"
    AI_GRADABILITY_RETINA = "fundus_retina_gradability"
    AI_VERIFICATION = "fundus_verification"

    @classmethod
    def list_columns(cls):
        columns = [member.value for column, member in cls.__members__.items()]
        return columns


class Status(str, Enum):
    ERROR = "error"
    QUEUED = "queued"
    DONE = "done"
    PROCESSING = "processing"
    EMPTY = "empty"


class Analysis:
    def __init__(
        self,
        analysis_id: str,
        cybersight_client: "CybersightAI",
        results: Optional[AnalysisResults] = None,
    ) -> None:
        """
        Initialize an Analysis instance.

        Args:
            analysis_id (str): The ID of the analysis.
            cybersight_client ('CybersightAI'): A reference to the Cybersight client.
            results (Optional[AnalysisResults]): The results of the analysis.
        """
        self.analysis_id: str = analysis_id
        self.cybersight_client = cybersight_client
        self._results = AnalysisResults(**results) if results else None
        self.tags = self._results.tags_string if self._results else None

    def get_status(self) -> str:
        """
        Retrieve the current status of the analysis.

        Returns:
            str: The current status of the analysis.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.cybersight_client.base_url}/analyses/{self.analysis_id}/"
            response = self.cybersight_client.make_request("GET", endpoint)
            return response.json().get("status")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to retrieve analysis status.") from e

    def get_results(
        self, poll: bool = False, interval: int = 5
    ) -> Optional[AnalysisResults]:
        """
        Retrieve the results of the analysis.

        Args:
            poll (bool):x   Whether to poll continuously until the analysis is done. Default is False.
            interval (int): The interval (in seconds) to wait between polling attempts. Default is 5 seconds.

        Returns:
            dict: The analysis results if the status is 'done', or None otherwise.

        Raises:
            Exception: If an error occurs during the request.
        """

        if self._results is not None and self._results.status in ["done", "error"]:
            return self._results

        max_poll_time = 300  # Maximum polling time in seconds (5 minutes)
        start_time = time.time()

        try:
            while True:
                status = self.get_status()
                if status == Status.DONE or status == Status.ERROR:
                    endpoint = f"{self.cybersight_client.base_url}/analyses/{self.analysis_id}/"
                    response = self.cybersight_client.make_request("GET", endpoint)
                    self._results = AnalysisResults(**response.json())
                    self.tags = self._results.tags_string
                    return self._results
                # elif status == Status.ERROR:
                #     raise Exception("Analysis encountered an error.")
                elif not poll:
                    return None
                else:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_poll_time:
                        raise Exception("Polling timed out after 5 minutes.")
                    logging.info(
                        f"Status: {status}. Polling again in {interval} seconds..."
                    )
                    time.sleep(interval)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to retrieve analysis results.") from e

    @property
    def results(self):
        return self.get_results()

    def parse_analysis_results(self):
        try:
            results = self.results

            row = {}

            row["tags"] = self.tags
            row["file_name"] = results.filename

            columns_names_mapping = {
                "dr_grading_score": "ICDR_value",
                "dr_grading": "ICDR_label",
                "referable_dr_nsc_score": "UK_NSC_value",
                "dr_grading_uk_nsc": "UK_NSC_label",
                "referable_dr_nsc_score": "referable_dr_UK_NSC_value",
                "referable_dr_nsc": "referable_dr_UK_NSC_label",
                "referable_dr_score_score": "referable_dr_value",
                "referable_dr_score": "referable_dr_label",
                "disc_anomaly_score": "disc_anomaly_value",
                "disc_anomaly": "disc_anomaly_label",
                "referable_glaucoma_score_score": "referable_glaucoma_value",
                "referable_glaucoma_score": "referable_glaucoma_label",
                "maculopathy_score": "maculopathy_value",
                "maculopathy": "maculopathy_label",
            }

            results_with_labels = [
                "dr_grading",
                "referable_dr_score",
                "referable_dr_nsc",
                "dr_grading_uk_nsc",
                "maculopathy",
                "disc_anomaly",
                "referable_glaucoma",
                "referable_glaucoma_score",
            ]

            results_with_scores = [
                "vcdr",
                "fundus_verification",
                "fundus_overall_gradability",
                # "fundus_macula_gradability",
                # "fundus_retina_gradability",
                # "fundus_disc_gradability",
            ]

            for result_with_label in results_with_labels:

                if list(
                    filter(lambda x: x.name == result_with_label, results.ai_outcomes)
                ):

                    try:
                        result_name_label = list(
                            filter(
                                lambda x: x.name == result_with_label,
                                results.ai_outcomes,
                            )
                        )[0]
                        row[f"{result_with_label}_score"] = np.argmax(
                            result_name_label.results[0].values
                        ).astype(int)
                        row[result_with_label] = result_name_label.results[0].labels[
                            np.argmax(result_name_label.results[0].values, axis=0)
                        ]

                    except:
                        if results.status == "done":
                            row[f"{result_with_label}_score"] = result_name_label.errors
                            row[result_with_label] = result_name_label.errors
                        else:
                            row[f"{result_with_label}_score"] = results.status
                            row[result_with_label] = results.status
                else:
                    continue

            for result_with_score in results_with_scores:

                if list(
                    filter(lambda x: x.name == result_with_score, results.ai_outcomes)
                ):

                    try:
                        result_name_score = list(
                            filter(
                                lambda x: x.name == result_with_score,
                                results.ai_outcomes,
                            )
                        )[0]
                        row[result_with_score] = result_name_score.results[0].values[0]

                    except:
                        if results.status == "done":
                            row[result_with_score] = result_name_score.errors
                        else:
                            row[result_with_score] = results.status

                else:
                    continue

            new_row = {
                (
                    columns_names_mapping[key] if key in columns_names_mapping else key
                ): value
                for key, value in row.items()
            }

            # return row
            return new_row

        except:
            raise Exception(
                "Failed to retrieve analysis results. Results may not be available yet."
            )


class StudyInput:
    def __init__(self, input_dict: Dict[str, Any]) -> None:
        """
        Initialize a StudyInput instance.

        Args:
            input_dict (Dict[str, Any]): A dictionary containing input attributes.
        """
        self.id: str = input_dict["id"]
        self.filename: str = input_dict["filename"]
        self.status: str = input_dict["status"]
        self.verified: bool = input_dict["verified"]
        self.gradable: bool = input_dict["gradable"]
        self.fov: str = input_dict["fov"]
        self.laterality: str = input_dict["laterality"]
        self.tags: Optional[Tags] = input_dict["tags"]

    def __repr__(self) -> str:
        """
        Return a string representation of the StudyInput instance.

        Returns:
            str: A string representation of the StudyInput.
        """
        return (
            f"StudyInput(id='{self.id}', filename='{self.filename}', status='{self.status}', "
            f"verified={self.verified}, gradable={self.gradable}, fov='{self.fov}', laterality='{self.laterality}', tags='{self.tags}))"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the StudyInput instance.

        Returns:
            str: A user-friendly string representation of the StudyInput.
        """
        return f"StudyInput: {self.id} - {self.filename} (Status: {self.status})"


class Config:
    def __init__(self, env: str):
        if env not in ["local", "dev", "prod", "za"]:
            raise ValueError(f"Invalid environment: {env}")

        self.base_url = {
            "local": "http://localhost:8000",
            "dev": "https://ai-core-dev.cybersight.org:443",
            "prod": "https://ai-core.cybersight.org:443",
            "za": "https://ai-core-south-africa.cybersight.org:443",
        }[env]

        self.consult_workspace_id = {
            "local": "7edaba82-e159-4d65-850c-5cbd687b00ad",
            "dev": "697437f7-d56c-4a6e-b17c-aea71e4e9d83",
            "prod": "7edaba82-e159-4d65-850c-5cbd687b00ad",
            "za": None,  # South Africa does not have a consult workspace
        }[env]


class Study:
    def __init__(
        self, study_dict: Dict[str, Any], cybersight_client: "CybersightAI"
    ) -> None:
        """
        Initialize a Study instance.

        Args:
            study_dict (Dict[str, Any]): A dictionary containing study attributes.
            cybersight_client ('CybersightAI'): A reference to the Cybersight client.
        """
        self.id: str = study_dict["id"]
        self.study_id: str = study_dict.get("study_id", "")
        self.workspace: Dict[str, Any] = study_dict.get("workspace", {})
        self.status: str = study_dict.get("status", "")
        self.created: str = study_dict.get("created", "")
        self.created_by: Dict[str, Any] = study_dict.get("created_by", {})
        self.study_metadata: List[Dict[str, Any]] = study_dict.get("study_metadata", [])
        self.verbose_outcomes: List[Dict[str, Any]] = study_dict.get(
            "verbose_outcomes", []
        )
        self.integration_data: List[Dict[str, Any]] = study_dict.get(
            "integration_data", []
        )
        self.analyses: List[Analysis[str, Any]] = [
            Analysis(
                analysis_id=x["id"], results=x, cybersight_client=cybersight_client
            )
            for x in study_dict.get("analyses", [])
        ]
        self.overviews: List[Dict[str, Any]] = study_dict.get("overviews", [])
        self.cybersight_client = (
            cybersight_client  # Store reference to the Cybersight client
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Study instance.

        Returns:
            str: A string representation of the Study.
        """
        return (
            f"Study(id='{self.id}', study_id='{self.study_id}', workspace={self.workspace}, "
            f"status='{self.status}', created='{self.created}', created_by={self.created_by}, "
            f"study_metadata={self.study_metadata}, verbose_outcomes={self.verbose_outcomes}, "
            f"integration_data={self.integration_data}, overviews={self.overviews})"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Study instance.

        Returns:
            str: A user-friendly string representation of the Study.
        """
        return f"Study: {self.id} - {self.study_id} (Status: {self.status}, Created: {self.created})"

    def create_analyses(
        self,
        study_inputs: Union[StudyInput, List[StudyInput]],
    ) -> Optional[Analysis]:
        """
        Creates an analysis for the study using StudyInput objects.

        Args:
            study_inputs (Union[StudyInput, List[StudyInput]]): Either a single StudyInput object or a list of StudyInput objects.

        Returns:
            Analysis: The created Analysis object.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:

            # TODO: Checks regarding number of images, etc...

            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.workspace['id']}/studies/{self.id}/analyses/"

            analyses_to_return = []

            for study_input in study_inputs:

                response = self.cybersight_client.make_request(
                    "POST",
                    endpoint,
                    data={
                        "input_item_id": study_input.id,
                        "tags": study_input.tags.value if study_input.tags else "",
                    },
                )

                analysis_data = response.json()
                analyses_to_return.append(
                    Analysis(analysis_data["id"], self.cybersight_client)
                )

            return analyses_to_return

        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to create analysis for study.") from e

    def get_study_analyses_results(self):
        """
        Returns all analyses for the study using analyses dictionaries to instantiate Analyses objects.

        Returns:
            Dict: dictionary of all analyses in specific study.

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            analysis_results = []

            for analysis in self.analyses:
                analysis_data = {}

                analysis_data["external_id"] = (
                    f"{self.study_metadata[0]['value']}/{self.study_id}/{analysis.analysis_id}"
                )

                analysis_data = {
                    **analysis_data,
                    **Analysis(
                        analysis.analysis_id,
                        cybersight_client=self.cybersight_client,
                    ).parse_analysis_results(),
                }

                analysis_results.append(analysis_data)

            return analysis_results

        except:
            raise Exception("Failed to retrieve analysis results for study.")


class Workspace:
    def __init__(
        self, workspace_dict: Dict[str, Any], cybersight_client: "CybersightAI"
    ) -> None:
        """
        Initialize a Workspace instance.

        Args:
            workspace_dict (Dict[str, Any]): A dictionary containing workspace attributes.
            cybersight_client ('Cybersight'): A reference to the Cybersight client.
        """
        self.id: str = workspace_dict["id"]
        self.protocol: str = workspace_dict["protocol"]
        self.name: str = workspace_dict["name"]
        self.reporting: bool = workspace_dict["reporting"]
        self.archived: bool = workspace_dict["archived"]
        self.description: str = workspace_dict["description"]
        self.protocol_details: Dict[str, Any] = workspace_dict.get(
            "protocol_details", {}
        )
        self.cybersight_client = cybersight_client

    def __repr__(self) -> str:
        """
        Return a string representation of the Workspace instance.

        Returns:
            str: A string representation of the Workspace.
        """
        return (
            f"Workspace(id='{self.id}', name='{self.name}', protocol='{self.protocol}', "
            f"reporting={self.reporting}, archived={self.archived}, description='{self.description}', "
            f"protocol_details={self.protocol_details})"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Workspace instance.

        Returns:
            str: A user-friendly string representation of the Workspace.
        """
        return f"Workspace: {self.id} - {self.name} (Protocol: {self.protocol}, Archived: {self.archived})"

    def get_allowed_tags(self) -> List[str]:
        """
        Retrieves the list of allowed tags from the protocol details.

        Returns:
            List[str]: A list of allowed tags.
        """
        allowed_tags = set()
        for input_detail in self.protocol_details.get("inputs", []):
            allowed_tags.update(input_detail.get("required_tags", []))
        return list(allowed_tags)

    def create_input_for_study(
        self, img_path: str, tags: Optional[Tags] = None
    ) -> Optional[StudyInput]:
        """
        Creates input for a study.

        Args:
            img_path (str): Path to the image file.
            tags (Optional[Tags]): Tags to associate with the input.

        Returns:
            StudyInput: An instance of StudyInput containing information about the created input, or None if an error occurs.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If an error occurs during the request.
        """
        try:

            allowed_tags = self.get_allowed_tags()

            if len(allowed_tags) and not tags:
                raise ValueError(
                    f"Tags are required for this workspace. Required tags: {', '.join(allowed_tags)}"
                )
            if tags:

                if not isinstance(tags, Tags):
                    raise ValueError("Tags must be a valid Tags enum value.")

                if len(allowed_tags):
                    # We only do tag validation if the workspace specifies said tags
                    tag_substring = tags.split(",")
                    invalid_tags = [
                        tag for tag in tag_substring if tag not in allowed_tags
                    ]
                    if invalid_tags:
                        raise ValueError(
                            f"Invalid tags provided: {', '.join(invalid_tags)}"
                        )
            # Check if the file exists
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"The file '{img_path}' does not exist.")

            endpoint = f"{self.cybersight_client.base_url}/inputs/"
            with open(img_path, "rb") as input_file:
                files = {"input_file": input_file}
                response = self.cybersight_client.make_request(
                    "POST", endpoint, files=files
                )
                response_data = response.json()
                response_data["tags"] = tags
            return StudyInput(response_data)
        except FileNotFoundError as e:
            logging.error(f"Error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to create input for study.") from e

    def create_study(self, **kwargs) -> Optional[Study]:
        """
        Adds a study to the workspace.

        Args:
            **kwargs: Arbitrary keyword arguments for metadata fields.

        Returns:
            Study: The created Study object if successful, None otherwise.

        Raises:
            ValueError: If required metadata fields are missing or invalid fields are provided.
            Exception: If an error occurs during the request.
        """
        required_metadata = [
            metadata["name"]
            for metadata in self.protocol_details.get("metadata", [])
            if not metadata["optional"]
        ]
        optional_metadata = [
            metadata["name"]
            for metadata in self.protocol_details.get("metadata", [])
            if metadata["optional"]
        ]

        # Validate and collect metadata from kwargs
        metadata = [{"name": key, "value": value} for key, value in kwargs.items()]

        # Check for missing required metadata
        missing_metadata = [req for req in required_metadata if req not in kwargs]
        if missing_metadata:
            raise ValueError(
                f"Missing required metadata: {', '.join(missing_metadata)}"
            )

        # Check for invalid metadata fields
        valid_metadata_fields = set(required_metadata + optional_metadata)
        invalid_metadata = [key for key in kwargs if key not in valid_metadata_fields]
        if invalid_metadata:
            raise ValueError(
                f"Invalid metadata fields provided: {', '.join(invalid_metadata)}"
            )

        try:
            payload = {"metadata": metadata}
            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.id}/studies"
            response = self.cybersight_client.make_request(
                "POST", endpoint, json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            return Study(response_data, self.cybersight_client)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to add study to workspace.") from e

    def get_studies(self, limit: int = 5000) -> Optional[List[Study]]:
        """
        Retrieves studies under the workspace with pagination.

        Args:
            limit (int): Number of studies to fetch per page (default is 100).

        Returns:
            List[Study]: A list of Study objects if studies are found, None otherwise.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.id}/studies/?include_analyses=true"
            studies = []
            next_page = endpoint  # Start with the initial endpoint

            while next_page:
                logging.info(f"Fetching studies from: {next_page}")
                response = self.cybersight_client.make_request(
                    "GET", next_page, params={"limit": limit}
                )
                response.raise_for_status()
                studies_data = response.json().get("items", [])
                studies.extend(
                    [
                        Study(study_data, self.cybersight_client)
                        for study_data in studies_data
                    ]
                )
                next_page = response.json().get(
                    "next"
                )  # Get next page URL if available

            return studies

        except requests.exceptions.RequestException as e:
            logging.error(f"Error: {e}")
            raise Exception("Failed to retrieve studies under the workspace.") from e

    def delete_workspace(self) -> Optional[Dict[str, Any]]:
        """
        Deletes the current workspace by its ID.

        Returns:
            dict: Response data containing information about the deleted workspace, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.id}"
            response = self.cybersight_client.make_request("DELETE", endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to delete workspace.") from e

    def add_user_to_workspace(
        self, user_id_to_add: str
    ) -> Optional[Dict[str, Union[str, str]]]:
        """
        Adds a user to the current workspace.

        Args:
            user_id_to_add (str): ID of the user to add.

        Returns:
            dict: Response data containing information about the added user, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.id}/users/{user_id_to_add}"
            response = self.cybersight_client.make_request("PUT", endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to add user to workspace.") from e

    def delete_study_from_workspace(self, study_id: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a study from the current workspace.

        Args:
            study_id (str): ID of the study to delete.

        Returns:
            dict: Response data containing information about the deleted study, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.cybersight_client.base_url}/workspaces/{self.id}/studies/{study_id}"
            response = self.cybersight_client.make_request("DELETE", endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to delete study from workspace.") from e

    def archive_workspace(self) -> Optional[Dict[str, Any]]:
        """
        Archives the current workspace.

        Returns:
            dict: Response data containing information about the archived workspace, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = (
                f"{self.cybersight_client.base_url}/workspaces/{self.id}/archived/true"
            )
            response = self.cybersight_client.make_request("PUT", endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to archive workspace.") from e


class CybersightAI:
    """
    A class for interacting with the Cybersight API.

    Attributes:
        base_url (str): The base URL of the Cybersight API.
        creds (dict): User credentials for authentication.
        access_token (str): Access token for authentication.
        headers (dict): Headers for API requests.
    """

    def __init__(self, email: str, password: str, env: str):
        """
        Initializes the Cybersight object by authenticating the user.

        Args:
            email (str): User email for authentication.
            password (str): User password for authentication.
            env (str): Environment to use ("local", "dev", "prod").
        """
        config = Config(env)
        self.base_url = config.base_url
        self.consult_workspace_id = config.consult_workspace_id
        self.creds: Dict[str, str] = {"email": email, "password": password}
        self._authenticate()

    def _authenticate(self):
        """
        Authenticates the user and sets the access token and headers.

        Raises:
            ValueError: If authentication fails.
        """
        try:
            response = requests.post(f"{self.base_url}/login", json=self.creds)
            response.raise_for_status()
            self.access_token = response.json().get("access_token")
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Authentication failed: {e}")
            raise ValueError("Authentication failed.") from e

    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Makes an HTTP request and retries if authentication fails.

        Args:
            method (str): The HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
            endpoint (str): The endpoint URL.
            **kwargs: Additional arguments to pass to the requests method.

        Returns:
            requests.Response: The response object.

        Raises:
            requests.exceptions.RequestException: If an error occurs during the request.
        """
        try:
            response = requests.request(
                method, endpoint, headers=self.headers, **kwargs
            )
            if response.status_code == 401:  # Unauthorized
                logging.info("Access token expired, re-authenticating...")
                self._authenticate()
                response = requests.request(
                    method, endpoint, headers=self.headers, **kwargs
                )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise

    def get_workspace_by_name(self, name: str) -> Optional[Workspace]:
        """
        Retrieves a workspace by its name.

        Args:
            name (str): Name of the workspace to retrieve.

        Returns:
            Workspace: The workspace object if found, None otherwise.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            workspaces = self.get_workspaces()
            for workspace in workspaces:
                if workspace.name == name:
                    workspace_id = workspace.id
                    response = self.make_request(
                        "GET", f"{self.base_url}/workspaces/{workspace_id}/"
                    )
                    workspace_full_data = response.json()
                    return Workspace(workspace_full_data, self)
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error: {e}")
            raise Exception("Failed to retrieve workspace by name.") from e

    def get_workspaces(
        self, include_personal: bool = True
    ) -> Optional[List[Workspace]]:
        """
        Retrieves all workspaces.

        Args:
            include_personal (bool): Whether to include personal workspaces. Defaults to True.

        Returns:
            Optional[List[Workspace]]: A list of Workspace objects if workspaces are found, None otherwise.

        Raises:
            ValueError: If the include_personal parameter is not a boolean.
            requests.exceptions.RequestException: If an error occurs during the request.
        """
        try:
            if not isinstance(include_personal, bool):
                raise ValueError("Parameter include_personal must be a boolean.")

            params = {"include_personal": include_personal}
            response = self.make_request(
                "GET", f"{self.base_url}/users/me/details", params=params
            )
            response_data = response.json()
            if isinstance(response_data, list):
                workspaces = [
                    Workspace(workspace_data, self) for workspace_data in response_data
                ]
            elif isinstance(response_data, dict):
                workspaces_data = response_data.get("workspaces", [])
                workspaces = [
                    Workspace(workspace_data, self)
                    for workspace_data in workspaces_data
                ]
            else:
                workspaces = []

            return workspaces
        except requests.exceptions.RequestException as e:
            logging.error(f"Error: {e}")
            raise Exception("Failed to retrieve workspaces.") from e

    def create_workspace(
        self, name: str, description: str, protocol: str, archived: bool = False
    ) -> Optional[Workspace]:
        """
        Creates a new workspace.

        Args:
            name (str): Name of the workspace.
            description (str): Description of the workspace.
            protocol (str): Protocol of the workspace.
            archived (bool): Whether the workspace is archived. Default is False.

        Returns:
            Workspace: The created workspace object, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            payload = {
                "name": name,
                "description": description,
                "protocol": protocol,
                "archived": archived,
            }
            response = self.make_request(
                "POST", f"{self.base_url}/workspaces/", json=payload
            )
            workspace_data = response.json()
            workspace_id = workspace_data["id"]
            response = self.make_request(
                "GET", f"{self.base_url}/workspaces/{workspace_id}/"
            )
            workspace_full_data = response.json()
            return Workspace(workspace_full_data, self)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to create workspace.") from e

    def process_image_list(self, image_paths: List[str]) -> List[StudyInput]:
        """
        Process a list of images and create inputs for a study.

        Args:
            image_paths (List[str]): A list of paths to the image files.

        Returns:
            List[StudyInput]: A list of created StudyInput objects.
        """
        try:
            study_inputs = []
            for image_path in image_paths:
                try:
                    study_input = self.create_input_for_study(image_path)
                    if study_input:
                        study_inputs.append(study_input)
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")

            if not study_inputs:
                raise Exception("No valid study inputs were created.")

            return study_inputs
        except Exception as e:
            logging.error(f"Failed to process image list: {e}")
            raise

    def create_programmatic_user(
        self, payload: Dict[str, Union[str, str]]
    ) -> Optional[Dict[str, Union[str, str]]]:
        """
        Creates a programmatic user.

        Args:
            payload (dict): Data for creating the user.

        Returns:
            dict: Response data containing information about the created user, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the request.
        """
        try:
            endpoint = f"{self.base_url}/users/programmatic/"
            response = self.make_request("POST", endpoint, json=payload)
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            raise Exception("Failed to create programmatic user.") from e
