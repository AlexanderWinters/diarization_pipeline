#!/usr/bin/env python3
import json
import sys
import argparse

def decode_unicode_in_json(input_data, output_file=None):
    """
    Decodes Unicode escape sequences in JSON data.

    Args:
        input_data: Either a JSON string or a file path.
        output_file: If provided, writes the decoded JSON to this file.
    """
    try:
        if isinstance(input_data, str) and input_data.endswith('.json'):
            with open(input_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(input_data)

        decoded_str = json.dumps(data, ensure_ascii=False, indent=2)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(decoded_str)
            print(f"Decoded JSON written to {output_file}")
        else:
            print(decoded_str)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Decode Unicode escape sequences in JSON files or strings."
    )
    parser.add_argument(
        'input',
        help='Input JSON file or string (use quotes for strings)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (optional, prints to stdout if omitted)'
    )
    args = parser.parse_args()

    decode_unicode_in_json(args.input, args.output)

if __name__ == "__main__":
    main()
