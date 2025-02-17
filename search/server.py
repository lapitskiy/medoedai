import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask, request, jsonify
from lstm_search import search  # Импортируем функцию search

app = Flask(__name__)


@app.route("/run-search", methods=["POST"])
def run_search():
    try:
        # Получаем параметры из запроса
        data = request.json
        query = data.get("query", "")
        logging.info(f"1) Received search request with query: {query}")

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Логируем перед вызовом search()
        logging.info("Calling search function...")
        results = search(query)
        logging.info(f"Search function returned: {results}")

        # Если код доходит сюда, но нет вывода в логах — проблема в jsonify()
        logging.info("Returning JSON response...")
        return jsonify({"message": "Parameter search completed!", "results": results})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)  # Логируем ошибку с трассировкой
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5052, debug=True)
