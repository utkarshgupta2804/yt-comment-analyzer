const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(express.json());
app.use(cors());

app.post("/analyze", async (req, res) => {
    try {
        const { url } = req.body;
        if (!url) {
            return res.status(400).json({ error: "Missing YouTube URL" });
        }

        // Call the Python API
        const response = await axios.post("http://127.0.0.1:5001/analyze", { url });

        res.json(response.data);
    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: "Server error" });
    }
});

app.listen(5000, () => {
    console.log("Node.js server running on port 5000");
});
