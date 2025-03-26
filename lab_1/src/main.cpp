#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>

class HebbNeuron {
private:
    std::vector<double> weights;
    std::string activation;
    double learningRate;
    bool normalizeWeights;

public:
    HebbNeuron(std::string act = "bipolar", double lr = 0.1, bool norm = false)
        : activation(act), learningRate(lr), normalizeWeights(norm) {
        weights.resize(25);
        for (auto& w : weights) {
            w = (static_cast<double>(rand()) / RAND_MAX * 2 - 1) * 0.1;
        }
    }

    int activate(double u) {
        if (activation == "binary") {
            return u >= 0 ? 1 : 0;
        } else if (activation == "bipolar") {
            return u >= 0 ? 1 : -1;
        } else if (activation == "sigmoid") {
            return 1 / (1 + std::exp(-u)) > 0.5 ? 1 : 0;
        } else if (activation == "tanh") {
            return std::tanh(u) > 0 ? 1 : -1;
        }
        return 0;
    }

    void trainHebb(const std::vector<int>& inputs) {
        double u = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            u += weights[i] * inputs[i];
        }
        int y = activate(u);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learningRate * inputs[i] * y;
        }

        if (normalizeWeights) {
            double norm = 0;
            for (auto w : weights) norm += w * w;
            norm = std::sqrt(norm);
            if (norm > 1e-6) {
                for (auto& w : weights) w /= norm;
            }
        }
    }

    int predict(const std::vector<int>& inputs) {
        double u = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            u += weights[i] * inputs[i];
        }
        return activate(u);
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(600, 600), "Neuron Training");
    HebbNeuron neuron("bipolar", 0.1, true);

    const int gridSize = 5;
    const int cellSize = 60;
    std::vector<std::vector<bool>> grid(gridSize, std::vector<bool>(gridSize, false));

    sf::Font font;
    if (!font.loadFromFile("../fonts/arial.ttf")) {
        std::cerr << "Error loading font" << std::endl;
        return -1;
    }

    sf::Text resultText("", font, 24);
    resultText.setPosition(50, 500);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            if (event.type == sf::Event::MouseButtonPressed) {
                int x = event.mouseButton.x;
                int y = event.mouseButton.y;

                if (x < gridSize * cellSize && y < gridSize * cellSize) {
                    int col = x / cellSize;
                    int row = y / cellSize;
                    grid[row][col] = !grid[row][col];
                }

                if (x > 400 && x < 560 && y > 50 && y < 100) {
                    std::vector<int> inputs;
                    for (int i = 0; i < gridSize; ++i) {
                        for (int j = 0; j < gridSize; ++j) {
                            inputs.push_back(grid[i][j] ? 1 : -1);
                        }
                    }
                    neuron.trainHebb(inputs);
                }

                if (x > 400 && x < 560 && y > 150 && y < 200) {
                    std::vector<int> inputs;
                    for (int i = 0; i < gridSize; ++i) {
                        for (int j = 0; j < gridSize; ++j) {
                            inputs.push_back(grid[i][j] ? 1 : -1);
                        }
                    }
                    int prediction = neuron.predict(inputs);
                    resultText.setString("Prediction: " + std::to_string(prediction));
                }

                if (x > 400 && x < 560 && y > 250 && y < 300) {
                    for (auto& row : grid) {
                        std::fill(row.begin(), row.end(), false);
                    }
                }
            }
        }

        window.clear(sf::Color::White);

        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
                cell.setPosition(j * cellSize, i * cellSize);
                cell.setOutlineThickness(2);
                cell.setOutlineColor(sf::Color::Black);
                cell.setFillColor(grid[i][j] ? sf::Color::Black : sf::Color::White);
                window.draw(cell);
            }
        }

        sf::RectangleShape trainButton(sf::Vector2f(150, 50));
        trainButton.setPosition(400, 50);
        trainButton.setFillColor(sf::Color::Green);
        sf::Text trainText("Train", font, 24);
        trainText.setPosition(440, 55);
        trainText.setFillColor(sf::Color::Black);

        sf::RectangleShape predictButton(sf::Vector2f(150, 50));
        predictButton.setPosition(400, 150);
        predictButton.setFillColor(sf::Color::Blue);
        sf::Text predictButtonText("Predict", font, 24);
        predictButtonText.setPosition(430, 155);
        predictButtonText.setFillColor(sf::Color::White);

        sf::RectangleShape clearButton(sf::Vector2f(150, 50));
        clearButton.setPosition(400, 250);
        clearButton.setFillColor(sf::Color::Red);
        sf::Text clearButtonText("Clear", font, 24);
        clearButtonText.setPosition(445, 255);
        clearButtonText.setFillColor(sf::Color::White);

        window.draw(trainButton);
        window.draw(trainText);
        window.draw(predictButton);
        window.draw(predictButtonText);
        window.draw(clearButton);
        window.draw(clearButtonText);
        window.draw(resultText);

        window.display();
    }

    return 0;
}