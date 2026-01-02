#!/bin/bash

###############################################################################
# Synthetic Data Migration Script
#
# This script automates the migration of synthetic data from the old project
# (Backup) to the new project structure, ensuring compatibility with MeZO
# training scripts and tasks.py data loader.
#
# Usage:
#   bash migrate_synthetic_data.sh [copy|link]
#
#   copy - Creates independent copies of data files (uses more disk space)
#   link - Creates symbolic links to original files (saves disk space)
#
# Default: link (if no argument provided)
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
PROJECT_ROOT="/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO"
OLD_DATA="/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data"
NEW_DATA="${PROJECT_ROOT}/Data/rejection_sampling/0_data"

# Migration mode (copy or link)
MODE="${1:-link}"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

###############################################################################
# Validation
###############################################################################

validate_environment() {
    print_header "Validating Environment"

    # Check if old data exists
    if [ ! -d "${OLD_DATA}" ]; then
        print_error "Old data directory not found: ${OLD_DATA}"
        exit 1
    fi
    print_success "Old data directory found"

    # Check if new project exists
    if [ ! -d "${PROJECT_ROOT}" ]; then
        print_error "New project directory not found: ${PROJECT_ROOT}"
        exit 1
    fi
    print_success "New project directory found"

    # Validate mode
    if [[ "${MODE}" != "copy" && "${MODE}" != "link" ]]; then
        print_error "Invalid mode: ${MODE}. Must be 'copy' or 'link'"
        exit 1
    fi
    print_success "Migration mode: ${MODE}"
}

###############################################################################
# Directory Creation
###############################################################################

create_directory_structure() {
    print_header "Creating Directory Structure"

    # Create base directory
    mkdir -p "${NEW_DATA}"
    print_success "Created: ${NEW_DATA}"

    # Create dataset directories
    local datasets=("Copa" "BOOLQ" "CB" "RTE" "ArcC_Cloze" "ArcC_MC")

    for dataset in "${datasets[@]}"; do
        mkdir -p "${NEW_DATA}/${dataset}"
        print_success "Created: ${NEW_DATA}/${dataset}"
    done
}

###############################################################################
# File Migration
###############################################################################

migrate_file() {
    local src="$1"
    local dst="$2"
    local file_description="$3"

    if [ ! -f "${src}" ]; then
        print_warning "Source file not found: ${src}"
        return 1
    fi

    if [ "${MODE}" == "copy" ]; then
        cp "${src}" "${dst}"
        print_success "Copied: ${file_description}"
    else
        ln -sf "${src}" "${dst}"
        print_success "Linked: ${file_description}"
    fi

    return 0
}

migrate_dataset() {
    local dataset_name="$1"
    shift
    local files=("$@")

    print_info "Migrating ${dataset_name} dataset..."

    local success_count=0
    local total_count=${#files[@]}

    for file in "${files[@]}"; do
        local src="${OLD_DATA}/${dataset_name}/${file}"
        local dst="${NEW_DATA}/${dataset_name}/${file}"

        if migrate_file "${src}" "${dst}" "${dataset_name}/${file}"; then
            ((success_count++))
        fi
    done

    echo -e "  ${success_count}/${total_count} files migrated for ${dataset_name}"
}

migrate_all_datasets() {
    print_header "Migrating Dataset Files"

    # Copa dataset
    migrate_dataset "Copa" \
        "copa_train.jsonl" \
        "copa_validation.jsonl" \
        "copa_test.jsonl"

    # BOOLQ dataset
    migrate_dataset "BOOLQ" \
        "boolq_train.jsonl" \
        "boolq_validation.jsonl"

    # CB dataset
    migrate_dataset "CB" \
        "cb_train.jsonl" \
        "cb_validation.jsonl" \
        "cb_test.jsonl"

    # RTE dataset
    migrate_dataset "RTE" \
        "rte_train.jsonl" \
        "rte_validation.jsonl" \
        "rte_test.jsonl"

    # ArcC_Cloze dataset
    migrate_dataset "ArcC_Cloze" \
        "ARC-Challenge_train.jsonl" \
        "ARC-Challenge_validation.jsonl" \
        "ARC-Challenge_test.jsonl"

    # ArcC_MC dataset (may not exist in old project)
    if [ -d "${OLD_DATA}/ArcC_MC" ]; then
        migrate_dataset "ArcC_MC" \
            "ARC-Challenge_train.jsonl" \
            "ARC-Challenge_validation.jsonl" \
            "ARC-Challenge_test.jsonl"
    else
        print_warning "ArcC_MC dataset not found in old project, skipping"
    fi
}

###############################################################################
# Verification
###############################################################################

verify_migration() {
    print_header "Verifying Migration"

    local datasets=("Copa" "BOOLQ" "CB" "RTE" "ArcC_Cloze")
    local required_files=(
        "Copa:copa_train.jsonl:copa_validation.jsonl"
        "BOOLQ:boolq_train.jsonl:boolq_validation.jsonl"
        "CB:cb_train.jsonl:cb_validation.jsonl"
        "RTE:rte_train.jsonl:rte_validation.jsonl"
        "ArcC_Cloze:ARC-Challenge_train.jsonl:ARC-Challenge_validation.jsonl"
    )

    local all_ok=true

    for entry in "${required_files[@]}"; do
        IFS=':' read -ra PARTS <<< "$entry"
        local dataset="${PARTS[0]}"
        local train_file="${PARTS[1]}"
        local valid_file="${PARTS[2]}"

        echo -e "\n${BLUE}Checking ${dataset}:${NC}"

        # Check train file
        if [ -f "${NEW_DATA}/${dataset}/${train_file}" ]; then
            local size=$(du -h "${NEW_DATA}/${dataset}/${train_file}" | cut -f1)
            print_success "${train_file} (${size})"
        else
            print_error "${train_file} - MISSING"
            all_ok=false
        fi

        # Check validation file
        if [ -f "${NEW_DATA}/${dataset}/${valid_file}" ]; then
            local size=$(du -h "${NEW_DATA}/${dataset}/${valid_file}" | cut -f1)
            print_success "${valid_file} (${size})"
        else
            print_error "${valid_file} - MISSING"
            all_ok=false
        fi
    done

    echo ""
    if [ "$all_ok" = true ]; then
        print_success "All required files are present"
        return 0
    else
        print_error "Some required files are missing"
        return 1
    fi
}

show_summary() {
    print_header "Migration Summary"

    echo -e "${BLUE}Directory Structure:${NC}"
    if command -v tree &> /dev/null; then
        tree -L 2 "${NEW_DATA}"
    else
        find "${NEW_DATA}" -maxdepth 2 -type f -name "*.jsonl" | sort
    fi

    echo -e "\n${BLUE}Storage Usage:${NC}"
    du -sh "${NEW_DATA}"

    echo -e "\n${BLUE}Migration Mode:${NC} ${MODE}"

    echo -e "\n${GREEN}Migration completed successfully!${NC}"
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "1. Verify the data structure: cd ${NEW_DATA} && tree -L 2"
    echo -e "2. Test with a training script from: ${PROJECT_ROOT}/running_scripts/"
    echo -e "3. The training scripts expect TASK path like:"
    echo -e "   TASK=${NEW_DATA}/Copa"
}

###############################################################################
# Main Execution
###############################################################################

main() {
    print_header "Synthetic Data Migration Tool"
    echo -e "Mode: ${MODE}"
    echo -e "Source: ${OLD_DATA}"
    echo -e "Destination: ${NEW_DATA}\n"

    # Ask for confirmation
    read -p "Proceed with migration? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Migration cancelled"
        exit 0
    fi

    # Execute migration steps
    validate_environment
    create_directory_structure
    migrate_all_datasets

    # Verify and show summary
    if verify_migration; then
        show_summary
        exit 0
    else
        print_error "Migration completed with errors. Please check the output above."
        exit 1
    fi
}

# Run main function
main
