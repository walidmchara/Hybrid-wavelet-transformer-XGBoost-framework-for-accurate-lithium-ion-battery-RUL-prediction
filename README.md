# Lithium-battery-analysis
Data Overview:
Four Li-ion batteries (# 5, 6, 7, and 18) underwent three operational profiles (charge, discharge, and impedance) at room temperature. Charging followed a constant current (CC) mode at 1.5A until reaching 4.2V, shifting to constant voltage (CV) until the current dropped to 20mA. Discharge involved a constant current (CC) at 2A until reaching specified voltage levels. Impedance measurements occurred through electrochemical impedance spectroscopy (EIS) with a frequency sweep from 0.1Hz to 5kHz. Repeated cycles accelerated battery aging, monitored by impedance changes. Experiments ceased upon reaching end-of-life (EOL) criteria—a 30% capacity fade from 2Ahr to 1.4Ahr. This dataset aids in predicting remaining charge for a given discharge cycle and remaining useful life (RUL).

File Structure:
- B0005.mat: Data for Battery #5
- B0006.mat: Data for Battery #6
- B0007.mat: Data for Battery #7
- B0018.mat: Data for Battery #18

Data Structure:
- **cycle:** Top-level structure array with charge, discharge, and impedance operations.
- **type:** Operation type (charge, discharge, or impedance).
- **ambient_temperature:** Ambient temperature in degrees Celsius.
- **time:** Date and time of the cycle start in MATLAB date vector format.
- **data:** Structure containing measurements:
  - For charge: 
    - Voltage_measured: Battery terminal voltage (Volts)
    - Current_measured: Battery output current (Amps)
    - Temperature_measured: Battery temperature (degrees Celsius)
    - Current_charge: Current measured at charger (Amps)
    - Voltage_charge: Voltage measured at charger (Volts)
    - Time: Time vector for the cycle (secs)
  - For discharge:
    - Voltage_measured: Battery terminal voltage (Volts)
    - Current_measured: Battery output current (Amps)
    - Temperature_measured: Battery temperature (degrees Celsius)
    - Current_charge: Current measured at load (Amps)
    - Voltage_charge: Voltage measured at load (Volts)
    - Time: Time vector for the cycle (secs)
    - Capacity: Battery capacity (Ahr) for discharge till 2.7V
  - For impedance:
    - Sense_current: Current in sense branch (Amps)
    - Battery_current: Current in battery branch (Amps)
    - Current_ratio: Ratio of the above currents
    - Battery_impedance: Battery impedance (Ohms) computed from raw data
    - Rectified_impedance: Calibrated and smoothed battery impedance (Ohms)
    - Re: Estimated electrolyte resistance (Ohms)
    - Rct: Estimated charge transfer resistance (Ohms)
