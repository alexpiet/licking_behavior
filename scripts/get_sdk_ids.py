import pandas as pd
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import PostgresQueryMixin
import importlib; importlib.reload(boa)

def get_ids():
    api = PostgresQueryMixin()
    query = '''
            SELECT
    
            oec.visual_behavior_experiment_container_id as container_id,
            oec.ophys_experiment_id,
            oe.workflow_state,
            d.full_genotype as full_genotype,
            d.id as donor_id,
            id.depth as imaging_depth,
            st.acronym as targeted_structure,
            os.name as session_name,
            equipment.name as equipment_name
    
            FROM ophys_experiments_visual_behavior_experiment_containers oec
            LEFT JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
            LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
            LEFT JOIN specimens sp ON sp.id=os.specimen_id
            LEFT JOIN donors d ON d.id=sp.donor_id
            LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
            LEFT JOIN structures st ON st.id=oe.targeted_structure_id
            LEFT JOIN equipment ON equipment.id=os.equipment_id
            '''
    
    experiment_df = pd.read_sql(query, api.get_connection())
    
    states_to_use = ['container_qc', 'passed']
   
    # Third pass
    subjects_to_use = [800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
     843387577, 789014546, 830940312, 834902192, 820871399, 827778598]
    
    conditions = [
        "workflow_state in @states_to_use",
        "equipment_name != 'MESO.1'",
        "donor_id in @subjects_to_use"
    ]
    
    query_string = ' and '.join(conditions)
    experiments_to_use = experiment_df.query(query_string)
    
    experiment_ids = experiments_to_use['ophys_experiment_id'].values
    return experiment_ids
