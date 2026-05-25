from backend.services.pdf_store import list_expired, delete

def delete_expired_pdfs():
    """
    Called every hour by the scheduler.
    Finds all PDFs older than 1 hour with keep=False and deletes them.
    """
    expired_ids = list_expired(max_age_hours=1)
    
    if not expired_ids:
        print("[cleanup] No expired PDFs to delete.")
        return
    
    print(f"[cleanup] Deleting {len(expired_ids)} expired PDF(s)...")
    for pdf_id in expired_ids:
        success = delete(pdf_id)
        status = "deleted" if success else "not found"
        print(f"[cleanup]   {pdf_id} → {status}")
