# Troubleshooting

Common issues and solutions for Job Card Extractor.

## Installation Issues

### "poppler is not found"

**Symptoms:** Error during PDF conversion or import failure.

**Solution:**

macOS:
```bash
brew install poppler
```

Linux (Ubuntu/Debian):
```bash
apt-get install poppler-utils
```

Windows: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

After installation, test:
```bash
pdftoppm --version
```

### Python Module Not Found

**Symptoms:** `ModuleNotFoundError: No module named 'cv2'` or similar.

**Solution:**

Ensure virtual environment is activated and dependencies installed:

```bash
# Activate venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### Memory Error During Installation

**Symptoms:** `MemoryError` when installing EasyOCR or large packages.

**Solution:**

Install packages individually with limited cache:

```bash
pip install --no-cache-dir opencv-python
pip install --no-cache-dir easyocr
pip install --no-cache-dir pyzbar
pip install --no-cache-dir pdf2image
```

## Processing Issues

### "PDF file not found"

**Symptoms:** FileNotFoundError when processing.

**Solution:**

1. Verify file exists:
   ```bash
   ls -la input.pdf
   ```

2. Use absolute path:
   ```bash
   python job_card_extractor.py /full/path/to/input.pdf -o output
   ```

3. Check file permissions (should be readable).

### "Corrupted PDF or unsupported format"

**Symptoms:** Error during PDF conversion or invalid image output.

**Solution:**

1. Verify PDF integrity with system tool:
   ```bash
   pdfinfo input.pdf
   ```

2. Try converting with another tool:
   ```bash
   pdftoppm -jpeg input.pdf test
   ```

3. If file is corrupted, try recovering:
   - Open in Adobe Reader and save as new PDF
   - Use online PDF repair tool

### Processing Hangs or Never Completes

**Symptoms:** Script runs but appears stuck; no output after hours.

**Solution:**

1. **Check system resources:**
   ```bash
   # macOS/Linux
   top -o %MEM | head -20
   ```
   If memory or CPU maxed, increase system resources or disable parallel processing.

2. **Disable parallel processing:**
   ```bash
   python job_card_extractor.py input.pdf -o output --no-parallel
   ```

3. **Use fast mode:**
   ```bash
   python job_card_extractor.py input.pdf -o output --fast-mode
   ```

4. **Set timeout and check logs:**
   ```bash
   timeout 600 python job_card_extractor.py input.pdf -o output
   ```
   Check `extraction_process_*.log` in output directory.

### "Out of Memory" Error

**Symptoms:** `MemoryError` or system becomes unresponsive during processing.

**Solution:**

1. **Disable parallel processing** (reduces memory by ~60%):
   ```bash
   python job_card_extractor.py large_file.pdf -o output --no-parallel
   ```

2. **Skip annotated images:**
   ```bash
   python job_card_extractor.py large_file.pdf -o output --no-annotated
   ```

3. **Split PDF into smaller chunks:**
   ```bash
   # Using command-line tool
   pdfseparate input.pdf part_%d.pdf

   # Process each part
   python job_card_extractor.py part_1.pdf -o output/
   python job_card_extractor.py part_2.pdf -o output/
   ```

4. **Increase system swap memory:**
   - macOS: System Preferences → Memory
   - Linux: Create swap file
   - Windows: Adjust virtual memory

## OCR Quality Issues

### Missing or Garbled Text

**Symptoms:** Operations not extracted, confidence scores are 0, logs show OCR failures.

**Solution:**

1. **Check annotated images** in `annotated/` directory to see if regions are detected.

2. **Try quality enhancement** (already default):
   ```bash
   python job_card_extractor.py input.pdf -o output  # No flags
   ```

3. **Add language support if needed:**
   ```bash
   # For French documents
   python job_card_extractor.py input.pdf -o output -l en fr
   ```

4. **Check PDF resolution:**
   - Low resolution PDFs may fail. Rescan or ask for higher quality.
   - Target: 200+ DPI

5. **Verify text is actually text** (not image):
   - Extract text with: `pdftotext input.pdf -`
   - If empty, PDF uses image-based text; may need different approach

### Low Confidence Scores

**Symptoms:** Extractions present but confidence < 0.5.

**Solution:**

1. **Check document quality:**
   - Blurry pages
   - Poor scan quality
   - Faded text

   Rescan document at higher resolution or quality settings.

2. **Verify barcode visibility:**
   - Check `annotated/` images for red barcode boxes
   - If barcodes not visible, document may be damaged

3. **Try without quality enhancement** (faster, sometimes more tolerant):
   ```bash
   python job_card_extractor.py input.pdf -o output --fast-mode
   ```

4. **Check logs** for specific failures:
   ```bash
   grep "confidence\|failed\|error" extraction_process_*.log
   ```

### "Language model not found"

**Symptoms:** Error when using non-English language codes.

**Solution:**

1. **EasyOCR downloads models automatically** on first use (200MB+ per language).

2. **Ensure internet connectivity** during first run.

3. **Pre-download models:**
   ```python
   import easyocr
   reader = easyocr.Reader(['en', 'fr'])  # Downloads if needed
   ```

4. **Check available languages:**
   ```python
   import easyocr
   print(easyocr.Reader.supported_languages)
   ```

## Barcode Detection Issues

### Barcodes Not Detected

**Symptoms:** Operations extracted but `extraction_strategy: "no_barcode_found"`.

**Solution:**

1. **Check barcode visibility** in `annotated/` images:
   - Red boxes show detected barcodes
   - If no red boxes, barcodes not detected

2. **Verify barcode quality:**
   - Barcodes must be clear and undamaged
   - Partial/faded barcodes won't scan

3. **Check barcode format:**
   - Tool supports: Code128, Code39, EAN, UPC, etc.
   - Some formats may not be recognized

4. **Try increasing preprocessing:**
   - Use default settings (quality enhancement on)
   - Avoid `--fast-mode`

5. **Check logs for barcode errors:**
   ```bash
   grep "barcode\|pyzbar" extraction_process_*.log
   ```

### Wrong Barcode Matched

**Symptoms:** Operations have barcodes but wrong ones (checked visually).

**Solution:**

1. **Review extraction strategy:**
   ```json
   {
     "op_number": "10",
     "extraction_strategy": "proximity_match"  // Fallback strategy
   }
   ```

2. **Verify document layout:**
   - Check `annotated/` images for area boundaries
   - If areas misaligned, detection fails

3. **Try without parallel processing:**
   - Sometimes parallel processing affects area detection
   ```bash
   python job_card_extractor.py input.pdf -o output --no-parallel
   ```

## Output Issues

### No Files Created in Output Directory

**Symptoms:** Output directory exists but empty after processing.

**Solution:**

1. **Check output directory permissions:**
   ```bash
   ls -ld output_dir
   # Should show: drwxr-xr-x
   ```

2. **Verify output directory path:**
   ```bash
   python job_card_extractor.py input.pdf -o "$(pwd)/output"
   ```

3. **Check for errors in logs:**
   - Look for `extraction_process_*.log` file
   - May be in current directory if output dir creation failed

4. **Try relative path:**
   ```bash
   python job_card_extractor.py input.pdf -o ./output
   ```

### JSON Files Are Empty

**Symptoms:** Output files created but contain `{}` or minimal data.

**Solution:**

1. **Check log file** for processing errors:
   ```bash
   tail -50 extraction_process_*.log
   ```

2. **Verify PDF has extractable content:**
   - Try: `pdftotext input.pdf -`
   - If empty, PDF may be image-only

3. **Run with verbose output** (check logs):
   ```bash
   python job_card_extractor.py input.pdf -o output 2>&1 | head -100
   ```

### Annotated Images Not Generated

**Symptoms:** `annotated/` directory missing or empty.

**Solution:**

1. **Check if intentionally skipped:**
   ```bash
   # These flags skip annotated images:
   python job_card_extractor.py input.pdf --no-annotated
   ```

2. **Verify output directory permissions:**
   ```bash
   chmod 755 output_dir
   ```

3. **Check disk space:**
   ```bash
   df -h  # Show available space
   ```
   Annotated images require ~500KB-1MB per page.

## Performance Issues

### Processing Very Slow

**Symptoms:** Single PDF takes hours to process.

**Solution:**

1. **Use fast mode:**
   ```bash
   python job_card_extractor.py input.pdf -o output --fast-mode
   ```
   Trades quality for ~3x speedup.

2. **Disable parallel processing** if system is resource-constrained:
   ```bash
   python job_card_extractor.py input.pdf -o output --no-parallel
   ```

3. **Reduce OCR languages:**
   ```bash
   python job_card_extractor.py input.pdf -o output -l en  # Default
   ```

4. **Skip intermediate files:**
   ```bash
   python job_card_extractor.py input.pdf -o output --no-raw --no-annotated
   ```

5. **Check system resources:**
   ```bash
   # Show CPU and memory usage
   top -u $USER
   ```
   If CPU < 20%, I/O may be bottleneck. Check disk speed.

### High Memory Usage (> 4GB)

**Symptoms:** System slows dramatically or crashes during processing.

**Solution:**

1. **Disable parallel processing immediately:**
   ```bash
   python job_card_extractor.py input.pdf -o output --no-parallel
   ```
   Reduces memory usage by 60%+.

2. **Split large PDFs:**
   ```bash
   pdfseparate input.pdf part_%d.pdf
   for f in part_*.pdf; do
     python job_card_extractor.py "$f" -o output --no-parallel
   done
   ```

3. **Skip annotated output:**
   ```bash
   python job_card_extractor.py input.pdf -o output --no-annotated
   ```

## Debugging

### Enable Detailed Logging

Check the `extraction_process_YYYYMMDD_HHMMSS.log` file for detailed processing information:

```bash
# Show recent errors
grep -i "error\|failed\|warning" extraction_process_*.log

# Show operation details
grep "\[OP-" extraction_process_*.log

# Show performance metrics
grep "time\|performance\|speed" extraction_process_*.log
```

### Analyze Annotated Images

1. **Open debug images:**
   ```bash
   open output/annotated/page_1_areas.jpg  # macOS
   display output/annotated/page_1_areas.jpg  # Linux
   ```

2. **Check for:**
   - Blue lines = detected area boundaries
   - Green boxes = OCR text regions
   - Red outlines = detected barcodes

3. **If areas misaligned:**
   - Document may have unusual layout
   - Try without parallel processing

### Check Raw Output

Examine `{filename}_raw.json` for intermediate extraction data:

```bash
cat output/*_raw.json | python -m json.tool | head -100
```

Look for:
- Detected barcodes
- OCR text samples
- Area coordinates

## Getting Help

1. **Check log file** for specific error messages
2. **Review example outputs** in `samples/` directory
3. **Examine annotated images** for visual debugging
4. **Check this guide** for your specific symptoms

If issue persists:
- Provide detailed error message from log
- Attach annotated image examples
- Describe document characteristics (resolution, format, etc.)

---

See also: [Architecture](architecture.md) | [User Guide](user-guide.md) | [API Reference](api-reference.md) | [Development](development.md)
