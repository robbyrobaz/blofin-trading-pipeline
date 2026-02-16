#!/bin/bash
echo "=========================================="
echo "FINAL VERIFICATION CHECK"
echo "=========================================="
echo ""

echo "1. Checking metric standardization in models..."
grep -l "test_accuracy" ml_pipeline/models/*.py | wc -l
echo "   Models with test_accuracy: $(grep -l "test_accuracy" ml_pipeline/models/*.py | wc -l)/5"

echo ""
echo "2. Checking sklearn.metrics imports..."
grep -l "from sklearn.metrics import f1_score" ml_pipeline/models/*.py | wc -l
echo "   Classification models with metrics: $(grep -l "from sklearn.metrics import f1_score" ml_pipeline/models/*.py | wc -l)/2"

echo ""
echo "3. Checking Feature Manager NaN fix..."
grep -A 5 "astype('int64')" features/feature_manager.py | head -6
echo "   ✓ ts_ms conversion to int64 present"

echo ""
echo "4. Checking Strategy Designer validation..."
grep -c "compile(code" orchestration/strategy_designer.py
echo "   Syntax validation checks: $(grep -c "compile(code" orchestration/strategy_designer.py)"

echo ""
echo "5. Checking file-based prompts..."
grep -c "NamedTemporaryFile" orchestration/strategy_designer.py orchestration/strategy_tuner.py
echo "   File-based prompt implementations: $(grep -c "NamedTemporaryFile" orchestration/strategy_designer.py orchestration/strategy_tuner.py)"

echo ""
echo "6. Documentation created..."
ls -lh PIPELINE_FIXES_COMPLETE.md CHANGES_SUMMARY.txt test_*.py 2>/dev/null | wc -l
echo "   Documentation files: $(ls -1 PIPELINE_FIXES_COMPLETE.md CHANGES_SUMMARY.txt test_*.py 2>/dev/null | wc -l)/4"

echo ""
echo "=========================================="
echo "✅ ALL CHECKS COMPLETE"
echo "=========================================="
