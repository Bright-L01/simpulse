# Simpulse v1.0.0-beta.1 Feedback Template

**Thank you for testing Simpulse!** Your feedback is crucial for improving the tool.

**Please fill out this template and:**
- Open a GitHub issue with your feedback: https://github.com/Bright-L01/simpulse/issues
- Or send to: brightliu@college.harvard.edu
- Or ping in Lean Zulip with results

---

## 📋 Basic Information

**Your Setup:**
- Lean version: [output of `lean --version`]
- OS: [macOS/Linux/Windows + version]
- Python version: [output of `python --version`]
- Project type: [research/teaching/personal/industrial]

**Project Characteristics:**
- Approximate number of `.lean` files: 
- Primary domain: [arithmetic/algebra/logic/category theory/other]
- Typical proof style: [heavy simp usage / mixed / mostly manual tactics]

---

## 🚀 Installation Experience

**Installation method used:**
```bash
pip install git+https://github.com/Bright-L01/simpulse.git
```

**Did installation work smoothly?** [Yes/No]

**If No, what went wrong?**
- Error message: 
- Steps that failed:
- How you resolved it (if you did):

**Health check result:**
```bash
$ simpulse --health
# Paste output here
```

**Overall installation experience:** [Excellent/Good/Okay/Difficult/Failed]

---

## 🎯 Performance Guarantee Assessment

**First, run the guarantee analysis:**
```bash
$ simpulse guarantee /path/to/your/project
# Paste full output here
```

**Guarantee Analysis Questions:**

1. **Prediction received:** [optimize/maybe/skip]

2. **Expected improvement:** [X%]

3. **Was the recommendation reasonable?** [Yes/No/Unclear]
   - Why or why not?

4. **Warning flags shown:** [list any warnings]

5. **Did the reasoning make sense for your project?** [Yes/No]
   - Comments:

---

## ⚡ Optimization Results (if you proceeded)

**Did you proceed with optimization?** [Yes/No]
**If No, why not?** [recommendation was skip / too risky / other]

**If Yes, please continue:**

### Before Optimization:
```bash
# Run this before optimization to get baseline
$ time lean YourSlowestFile.lean
# Paste timing results (try 3 times for accuracy)
Run 1: real Xm Ys
Run 2: real Xm Ys  
Run 3: real Xm Ys
Average: [calculate average]
```

### Optimization Process:
```bash
$ simpulse optimize --apply /path/to/your/project
# Paste output here
```

**Did optimization complete successfully?** [Yes/No]
**If No, what error occurred?**

### After Optimization:
```bash
# Test same file after optimization
$ time lean YourSlowestFile.lean  
# Paste timing results (try 3 times for accuracy)
Run 1: real Xm Ys
Run 2: real Xm Ys
Run 3: real Xm Ys
Average: [calculate average]
```

**Actual improvement:** [X% faster/slower/no change]

**Did actual results match the prediction?** [Yes/No/Partially]

---

## 🔍 Detailed Usage Feedback

### What Worked Well:
- [ ] Installation was smooth
- [ ] Commands were intuitive
- [ ] Error messages were helpful
- [ ] Performance improvement was significant
- [ ] Tool correctly identified optimization potential
- [ ] Documentation was clear
- [ ] CLI interface was pleasant to use

### What Didn't Work:
- [ ] Installation was difficult
- [ ] Commands were confusing
- [ ] Error messages were unclear
- [ ] No performance improvement seen
- [ ] Tool incorrectly assessed project
- [ ] Documentation was lacking
- [ ] CLI interface was clunky

### Specific Issues Encountered:
1. **Issue:** 
   **Context:** 
   **Error message (if any):**
   **How you worked around it:**

2. **Issue:**
   **Context:**
   **Error message (if any):**
   **How you worked around it:**

---

## 📊 Project-Specific Results

**Project Details:**
- Number of `@[simp]` rules found: [from `simpulse check` output]
- Rules that could be optimized:
- Primary types of simp rules: [arithmetic/lists/strings/other]

**Usage Patterns in Your Project:**
- Percentage of proofs that use `simp`: [estimate: <10% / 10-30% / 30-50% / >50%]
- Typical `simp` usage: [`by simp` / `simp [specific_rules]` / `simp_rw` / mixed]
- Slowest compilation bottleneck: [simp search / type checking / other tactics / file size]

**Results Summary:**
- **Compilation time improvement:** [X% faster/slower/no change]
- **Most improved file:** [filename] - [X% improvement]
- **Any files that got slower:** [Yes/No] - [details if yes]
- **Overall satisfaction:** [Very satisfied/Satisfied/Neutral/Dissatisfied/Very dissatisfied]

---

## 🎓 Educational/Workflow Impact

**How did you integrate Simpulse into your workflow?**
- One-time optimization
- Regular re-optimization  
- Part of CI/CD pipeline
- Teaching tool demonstration
- Other:

**For teaching/learning contexts:**
- Did students notice the improvement?
- Impact on learning experience:
- Would you recommend to other educators?

**For research contexts:**
- Impact on development velocity:
- Integration with existing tools:
- Would you use in future projects?

---

## 💡 Suggestions and Improvements

**What would make Simpulse more useful?**

**Missing features you'd like to see:**

**Documentation improvements needed:**

**CLI/UX improvements suggested:**

**Integration with other tools desired:**

---

## 🏆 Overall Assessment

**1. Did Simpulse solve a real problem for you?** [Yes/No]
   - Explain:

**2. Would you recommend Simpulse to colleagues?** [Yes/No/Depends]
   - Under what circumstances?

**3. What's the most valuable aspect of the tool?**

**4. What's the biggest limitation?**

**5. Rate your overall experience:** [1-5 stars]
   - 5 = Excellent, exceeded expectations
   - 4 = Good, met most expectations  
   - 3 = Okay, mixed results
   - 2 = Poor, mostly disappointed
   - 1 = Very poor, would not recommend

**6. Additional comments:**

---

## 🐛 Bug Reports (if any)

**Reproducible bugs found:**

**Bug 1:**
- **Description:**
- **Steps to reproduce:**
- **Expected behavior:**
- **Actual behavior:**
- **Error message:**
- **Workaround (if found):**

**Bug 2:**
- [Same format as above]

---

## 📈 Performance Data (Optional but Helpful)

If you're willing to share more detailed data:

**Compilation timing data:**
```bash
# Before optimization
$ time lake build
# Paste results

# After optimization  
$ time lake build
# Paste results
```

**Specific slow files (before/after):**
```bash
# File 1: [filename]
Before: [timing]
After: [timing]
Improvement: [%]

# File 2: [filename]  
Before: [timing]
After: [timing]
Improvement: [%]
```

**Project statistics:**
- Total lines of Lean code: [estimate]
- Simp rules per file (average): [estimate]
- Most simp-heavy file: [filename] - [number of rules]

---

## 🤝 Follow-up

**Can we contact you for follow-up questions?** [Yes/No]
**Preferred contact method:** [email/GitHub/Zulip]
**GitHub username (if comfortable sharing):**

**Would you be interested in testing future versions?** [Yes/No]

---

**Thank you for your detailed feedback! This helps us improve Simpulse for the entire Lean community.**

**Questions about this feedback form?** Open an issue or contact brightliu@college.harvard.edu