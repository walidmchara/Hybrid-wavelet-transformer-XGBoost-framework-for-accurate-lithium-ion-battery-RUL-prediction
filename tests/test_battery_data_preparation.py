"""Tests run without downloading the large NASA/CALCE files."""
from pathlib import Path
import io
import math
import sys
import zipfile
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_battery_data import _finish, _parse_calce_workbook, load_nasa_mat


def test_soh_is_normalized_to_first_observation():
    df=_finish(pd.DataFrame({'capacity_ah':[1.5,1.2,1.0]}),'B0005',nominal_ah=2.0)
    assert list(np.round(df['target'],3))==[1,.8,.667]
    assert list(df['remaining_observed_cycles'])==[2,1,0]
    assert list(df['soh_nominal'])==[.75,.6,.5]


def test_calce_cumulative_capacity_must_be_differenced():
    from openpyxl import Workbook
    from datetime import datetime, timedelta
    wb=Workbook();ws=wb.active;ws.title='Channel_1-011'
    ws.append(['Data_Point','Test_Time(s)','Date_Time','Cycle_Index','Current(A)','Voltage(V)','Discharge_Capacity(Ah)'])
    start=datetime(2010,9,1)
    # Cycle 2 starts at cumulative discharge 1.0 Ah, not at 0 Ah.
    for row in [(1,1,-.5,3.9,0.0),(1,2,-.5,3.8,.4),(1,3,-.5,3.7,1.0),
                (2,4,-.5,3.9,1.0),(2,5,-.5,3.8,1.4),(2,6,-.5,3.7,1.8)]:
        cycle,t,current,volt,cap=row
        ws.append([t,t*10,start+timedelta(seconds=t*10),cycle,current,volt,cap])
    buf=io.BytesIO();wb.save(buf)
    result=_parse_calce_workbook(buf.getvalue(),'CS2_35_mock.xlsx')
    assert len(result)==2
    assert math.isclose(result[0]['capacity_ah'],1.0,abs_tol=1e-8)
    assert math.isclose(result[1]['capacity_ah'],0.8,abs_tol=1e-8)
    assert result[1]['discharge_duration_s']==20
