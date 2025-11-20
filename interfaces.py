"""
I/O for GACODE input.gacode and output.gacode files.

This module provides a lightweight, numpy-only equivalent to MITIM's
PROFILES_GACODE and MINTtransform.gacode_to_powerstate, without external
dependencies. It also offers convenience functions to read/write PlasmaState.
"""

import numpy as np
import copy
from typing import Dict, List, Optional
from collections import OrderedDict
#from state import PlasmaState
from tools import io


class gacode:
    """A minimal PROFILES_GACODE-like container

    Attributes
    ----------
    profiles: Dict[str, np.ndarray or float]
        Flat dictionary of parsed arrays and scalars from GACODE file.
    derived: Dict[str, np.ndarray or float]
        Minimal derived quantities computed from profiles.
    species: List[Dict]
        List of species dictionaries with keys name, Z, A, density (coarse grid).
    """

    def __init__(self, filepath: Optional[str] = None, profiles: Optional[Dict[str, np.ndarray]] = None):

        self.titles_singleNum = ["nexp", "nion", "shot", "name", "type", "time"]
        self.titles_singleArr = [
            "masse",
            "mass",
            "ze",
            "z",
            "torfluxa(Wb/radian)",
            "rcentr(m)",
            "bcentr(T)",
            "current(MA)",
        ]
        self.titles_single = self.titles_singleNum + self.titles_singleArr

        self.file = filepath
        self.header: List[str] = []
        self.profiles: Dict[str, np.ndarray] = OrderedDict()

        if profiles is not None:
            # Construct from in-memory profiles
            self.profiles = OrderedDict((k, copy.deepcopy(v)) for k, v in profiles.items())
        elif self.file is not None:
            with open(self.file, "r") as f:
                self.lines = f.readlines()
            # Read file and store raw data
            self.readHeader()  # Optional; not all files include a header
            self.readProfiles()
            self.header = getattr(self, "header", [])

        # Build simple base-name mapping if profiles exist
        if hasattr(self, "profiles") and len(self.profiles) > 0:
            self.profiles_mapping = {key: key.split('(')[0] for key in self.profiles.keys()}
        else:
            self.profiles_mapping = {}
    

    def write(self, file=None, limitedNames=False):
        print("\t- Writting input.gacode file")

        if file is None:
            file = self.file

        with open(file, "w") as f:
            for line in self.header:
                f.write(line)

            for i in self.profiles:
                if "(" not in i:
                    f.write(f"# {i}\n")
                else:
                    f.write(f"# {i.split('(')[0]} | {i.split('(')[-1].split(')')[0]}\n")

                if i in self.titles_single:
                    if i == "name" and limitedNames:
                        newlist = [self.profiles[i][0]]
                        for k in self.profiles[i][1:]:
                            if k not in [
                                "D",
                                "H",
                                "T",
                                "He4",
                                "he4",
                                "C",
                                "O",
                                "Ar",
                                "W",
                            ]:
                                newlist.append("C")
                            else:
                                newlist.append(k)
                        print(
                            f"\n\n!! Correcting ion names from {self.profiles[i]} to {newlist} to avoid TGYRO radiation error (to solve in future?)\n\n",
                            typeMsg="w",
                        )
                        listWrite = newlist
                    else:
                        listWrite = self.profiles[i]

                    if io.isfloat(listWrite[0]):
                        listWrite = [f"{i:.7e}".rjust(14) for i in listWrite]
                        f.write(f"{''.join(listWrite)}\n")
                    else:
                        f.write(f"{' '.join(listWrite)}\n")

                else:
                    if len(self.profiles[i].shape) == 1:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            valt = f"{round(val,99):.7e}".rjust(15)
                            f.write(f"{pos}{valt}\n")
                    else:
                        for j, val in enumerate(self.profiles[i]):
                            pos = f"{j + 1}".rjust(3)
                            txt = "".join([f"{k:.7e}".rjust(15) for k in val])
                            f.write(f"{pos}{txt}\n")

        print(f"\t\t~ File {io.clipstr(file)} written")

        # Update file
        self.file = file

    # # --- conversion methods ---
    # def to_state(self) -> "PlasmaState":
    #     """Convert this GACODE object into a PlasmaState."""
    #     from .state import PlasmaState
    #     return PlasmaState.from_gacode(self)

    # @classmethod
    # def from_state(cls, state: "PlasmaState") -> "gacode":
    #     """Rebuild a minimal GACODE object from a PlasmaState."""
    #     return state.to_gacode()



    # *****************

    def readHeader(self):
        for i in range(len(self.lines)):
            if "# nexp" in self.lines[i]:
                istartProfs = i
        self.header = self.lines[:istartProfs]

    def readProfiles(self):
        singleLine, title, var = None, None, None  # for ruff complaints

        # ---
        found = False
        self.profiles = OrderedDict()
        for i in range(len(self.lines)):
            if self.lines[i][0] == "#" and self.lines[i + 1][0] != "#":
                # previous
                if found and not singleLine:
                    self.profiles[title] = np.array(var)
                    if self.profiles[title].shape[1] == 1:
                        self.profiles[title] = self.profiles[title][:, 0]

                linebr = self.lines[i].split("#")[1].split("\n")[0].split()
                title_Orig = linebr[0]
                if len(linebr) > 1:
                    unit = self.lines[i].split("#")[1].split("\n")[0].split()[2]
                    title = title_Orig + f"({unit})"
                else:
                    title = title_Orig
                found, var = True, []

                if title in self.titles_single:
                    singleLine = True
                else:
                    singleLine = False
            elif found:
                var0 = self.lines[i].split()
                if singleLine:
                    if title in self.titles_singleArr:
                        self.profiles[title] = np.array([float(i) for i in var0])
                    else:
                        self.profiles[title] = np.array(var0)
                else:
                    # varT = [float(j) for j in var0[1:]]
                    """
                    Sometimes there's a bug in TGYRO, where the powers may be too low (E-191) that cannot be properly written
                    """
                    varT = [
                        float(j) if (j[-4].upper() == "E" or "." in j) else 0.0
                        for j in var0[1:]
                    ]

                    var.append(varT)

        # last
        if not singleLine:
            while len(var[-1]) < 1:
                var = var[:-1]  # Sometimes there's an extra space, remove
            self.profiles[title] = np.array(var)
            if self.profiles[title].shape[1] == 1:
                self.profiles[title] = self.profiles[title][:, 0]

        # Accept omega0
        if ("w0(rad/s)" not in self.profiles) and ("omega0(rad/s)" in self.profiles):
            self.profiles["w0(rad/s)"] = self.profiles["omega0(rad/s)"]
            del self.profiles["omega0(rad/s)"]

        ## Insert n0
        # if ("n0(10^19/m^3)" not in self.profiles):
        #     self.profiles["n0(10^19/m^3)"] = self.profiles["ni(10^19/m^3)"]*1e-6

        ## Insert static wtor
        # if ("wtor(rad/s)" not in self.profiles):
        #     self.profiles["wtor(rad/s)"] = self.profiles["w0(rad/s)"]

        