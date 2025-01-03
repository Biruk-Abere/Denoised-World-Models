import os
import argparse
import glob
import tensorflow as tf
import jsonlines

# This function retrieves a specific value from a summary based on its tag.
# It checks for various types of values such as simple_value, int64_value, etc.
def get_summary_value(summary, tag):
    for value in summary.value:
        if value.tag == tag:
            if value.HasField('simple_value'):
                return value.simple_value
            elif value.HasField('int64_value'):
                return value.int64_value
            elif value.HasField('bool_value'):
                return value.bool_value
            elif value.HasField('string_value'):
                return value.string_value
            elif value.HasField('histo'):
                return value.histo
    return None

# This function takes a TensorFlow event file as input and writes its content to a JSONL file.
# The function also allows filtering of data based on a tag.
def convert_event_to_jsonl(event_file, jsonl_file, tag):
    step_values = {}
    for e in tf.compat.v1.train.summary_iterator(event_file):
        if e.summary is not None:
            value = get_summary_value(e.summary, tag)
            if value is not None:
                step_values[e.step] = (e.wall_time, value)

    with jsonlines.open(jsonl_file, mode='w') as writer:
        for step, (wall_time, value) in step_values.items():
            writer.write({"step": step, "wall_time": wall_time, "value": value})

# This function finds all TensorFlow event files in a folder and converts the first found file to JSONL.
def convert_all_event_files(tag, folder_name):
    event_files = glob.glob(os.path.join(folder_name, 'events.out.tfevents.*'))
    if event_files:
        event_file = event_files[0]
        jsonl_file = os.path.join(folder_name + '/' +folder_name.rstrip('/').split('/')[-1] + '.jsonl')
        print(jsonl_file)  # prints the destination JSONL file name (for debugging or verification purposes)
        convert_event_to_jsonl(event_file, jsonl_file, tag)

# Main function that parses command line arguments and initiates the file conversion process.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True)  # Folder name containing event files
    parser.add_argument("--tag", type=str, required=True)  # Tag to filter data in the event files
    args = parser.parse_args()

    convert_all_event_files(args.tag, args.folder_name)

# If this script is run as a standalone program, it starts the main function.
if __name__ == "__main__":
    main()