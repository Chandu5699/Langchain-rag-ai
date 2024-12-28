def process_files_in_parallel(file_paths):
    try:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for file_path in file_paths:
                file_type = file_path.split('.')[-1]
                executor.submit(process_file, file_path, file_type)
    except Exception as e:
        logging.error(f"Error processing files in parallel: {e}")


def process_and_chunk_data(batch_size=1000):
    try:
        data_batch = []
        for record in load_data_in_batches(batch_size):
            content = None
            metadata = {
                "id": record.get("_id"),
                "type": record.get("type"),
                "source": record.get("source"),
                "timestamp": record.get("timestamp"),
            }
            
            if record.get("type") == "pdf":
                content = extract_text_from_pdf(record.get("file_path"))
            
            elif record.get("type") == "json":
                content = record.get("content")
            
            elif record.get("type") == "audio":
                content = extract_text_from_audio(record.get("file_path"))
            elif record.get("type") == "Jira":
                content = extract_text_from_video(record.get("file_path"))
            elif record.get("type") == "Confluence":
                content = extract_text_from_video(record.get("file_path"))
            
            data_batch.append((content, metadata))
            
            if len(data_batch) == batch_size:
                yield data_batch
                data_batch = []
        
        if data_batch:
            yield data_batch
    except Exception as e:
        logging.error(f"Error in processing and chunking data: {e}")
        raise