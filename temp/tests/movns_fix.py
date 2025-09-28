"""
Fix for MOVNS performance issue
Process single solution per iteration instead of entire archive
"""

import sys
import os

sys.path.append('E:/pycommend/pycommend-code')

# Read the file
with open('E:/pycommend/pycommend-code/src/optimizer/movns_vns.py', 'r') as f:
    lines = f.readlines()

# Find the problematic section (around line 497-524)
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'for iteration in range(self.max_iterations):' in line:
        start_idx = i
    if start_idx and 'self.truncate_archive()' in line:
        end_idx = i + 1
        break

if start_idx and end_idx:
    # Replace the problematic section
    new_section = """        for iteration in range(self.max_iterations):
            improved = False

            adaptive_samples = min(5 + iteration // 3, 12)

            # Select one solution from archive (standard VNS approach)
            if len(self.archive) > 0:
                sol_dict = random.choice(self.archive)
                solution = sol_dict['chromosome']
            else:
                solution = self.smart_initialization('hybrid')

            k = 0

            # VNS loop for single solution
            while k < self.k_max:
                # Shaking phase
                s_prime = self.shake(solution, self.neighborhoods[k], intensity=k+1)

                # Local search with MOBI/P
                improved_solutions = self.mobi_p_local_search(s_prime, self.neighborhoods[k], samples=adaptive_samples)

                # Update archive with improved solutions
                archive_updated = False
                for (new_sol, new_obj) in improved_solutions:
                    if self.update_archive(new_sol, new_obj):
                        archive_updated = True
                        improved = True
                        solution = new_sol  # Continue from improved solution

                # Neighborhood change strategy
                if archive_updated:
                    k = 0  # Restart from first neighborhood
                else:
                    k += 1  # Move to next neighborhood

            self.truncate_archive()
"""

    # Write back the fixed version
    lines[start_idx:end_idx] = new_section

    with open('E:/pycommend/pycommend-code/src/optimizer/movns_vns.py', 'w') as f:
        f.writelines(lines)

    print(f"Fixed MOVNS performance issue")
    print(f"Replaced lines {start_idx+1} to {end_idx+1}")
    print("Now processes single solution per iteration (standard VNS)")
else:
    print("Could not find section to replace")