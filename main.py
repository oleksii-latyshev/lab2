import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def define_variables():
    distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance') 
    speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')      
    road_condition = ctrl.Antecedent(np.arange(0, 11, 1), 'road_condition') 
    throttle = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'throttle') 

    action = ctrl.Consequent(np.arange(0, 101, 1), 'action')

    return distance, speed, road_condition, throttle, action

def define_membership_functions(distance, speed, road_condition, throttle, action):
    distance['close'] = fuzz.trimf(distance.universe, [0, 0, 50])
    distance['medium'] = fuzz.trimf(distance.universe, [0, 50, 100])
    distance['far'] = fuzz.trimf(distance.universe, [50, 100, 100])

    speed['low'] = fuzz.trimf(speed.universe, [0, 0, 50])
    speed['medium'] = fuzz.trimf(speed.universe, [0, 50, 100])
    speed['high'] = fuzz.trimf(speed.universe, [50, 100, 100])

    road_condition['poor'] = fuzz.trimf(road_condition.universe, [0, 0, 5])
    road_condition['normal'] = fuzz.trimf(road_condition.universe, [0, 5, 10])
    road_condition['good'] = fuzz.trimf(road_condition.universe, [5, 10, 10])

    throttle['low'] = fuzz.trimf(throttle.universe, [0, 0, 0.5])
    throttle['medium'] = fuzz.trimf(throttle.universe, [0, 0.5, 1])
    throttle['high'] = fuzz.trimf(throttle.universe, [0.5, 1, 1])

    action['brake'] = fuzz.trimf(action.universe, [0, 0, 50])
    action['hold'] = fuzz.trimf(action.universe, [0, 50, 100])
    action['accelerate'] = fuzz.trimf(action.universe, [50, 100, 100])

def define_rules(distance, speed, road_condition, throttle, action):
    rules = [
        ctrl.Rule(distance['close'] & speed['high'], action['brake']),
        ctrl.Rule(distance['close'] & speed['medium'], action['brake']),
        ctrl.Rule(distance['medium'] & speed['high'], action['brake']),
        ctrl.Rule(distance['medium'] & speed['medium'] & road_condition['poor'], action['brake']),
        ctrl.Rule(distance['medium'] & speed['medium'] & road_condition['normal'], action['hold']),
        ctrl.Rule(distance['far'] & speed['low'], action['accelerate']),
        ctrl.Rule(distance['far'] & speed['medium'], action['accelerate']),
        ctrl.Rule(distance['far'] & speed['high'], action['hold']),
        
        ctrl.Rule(road_condition['poor'], action['brake']),
        ctrl.Rule(road_condition['normal'], action['hold']),
        ctrl.Rule(road_condition['good'] & throttle['high'], action['accelerate']),
        
        ctrl.Rule(distance['close'] & road_condition['poor'], action['brake']),
        ctrl.Rule(distance['medium'] & road_condition['good'], action['hold']),
        ctrl.Rule(distance['far'] & road_condition['poor'], action['hold']),
        ctrl.Rule(distance['far'] & road_condition['good'], action['accelerate']),
        
        ctrl.Rule(speed['low'] & throttle['low'], action['hold']),
        ctrl.Rule(speed['medium'] & throttle['medium'], action['hold']),
        ctrl.Rule(speed['high'] & throttle['high'], action['hold']),
        
        ctrl.Rule(speed['low'] & road_condition['good'], action['accelerate']),
        ctrl.Rule(speed['high'] & road_condition['poor'], action['brake']),
    ]
    return rules

def process_data(file_path):
    data = pd.read_csv(file_path)
    
    np.random.seed(42)
    data['road_condition'] = np.random.randint(0, 11, size=len(data))
    
    return data

def plot_action_counts(action_counts):
    plt.bar(action_counts.keys(), action_counts.values(), color=['red', 'yellow', 'green'])
    plt.title('Count of driver actions')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.show()

def run_simulation(rules, distance, speed, road_condition, throttle, action, data):
    control_system = ctrl.ControlSystem(rules)
    simulator = ctrl.ControlSystemSimulation(control_system)
    
    action_counts = {"brake": 0, "hold": 0, "accelerate": 0}
    action_labels = {
        "brake": (0, 40),
        "hold": (40, 50),
        "accelerate": (50, 100),
    }
    
    results = [] 
    
    for index, row in data.iterrows():
        simulator.input['distance'] = np.random.uniform(0, 100)
        simulator.input['speed'] = row['speed'] * 3.6  # Conversion to km/h
        simulator.input['road_condition'] = row['road_condition']
        simulator.input['throttle'] = row['throttle']
        
        simulator.compute()
        action_value = simulator.output['action']
        
        action_text = None
        for label, (low, high) in action_labels.items():
            if low <= action_value <= high:
                action_text = label
                action_counts[label] += 1
                break
        
        results.append(action_text)
        print(f"⚙️ Row {index + 1}: Action = {action_text} ({action_value:.2f})")
    
    plot_action_counts(action_counts)

def main():
    file_path = 'dataset/driving_log.csv' 
    data = process_data(file_path)
    
    distance, speed, road_condition, throttle, action = define_variables()

    define_membership_functions(distance, speed, road_condition, throttle, action)

    rules = define_rules(distance, speed, road_condition, throttle, action)

    run_simulation(rules, distance, speed, road_condition, throttle, action, data)

if __name__ == "__main__":
    main()
