# Neural Network From Scratch

This repository is a personal sandbox for understanding the building blocks of deep learning. I am rebuilding a tiny, highly simplified version of libraries like PyTorch to explore how tensors, automatic differentiation, and neural network layers operate under the hood.

## Why This Exists
- Practice translating theoretical concepts into working code.
- Experiment with the internals of tensor math and autograd without the abstractions of a production framework.
- Share a readable codebase for others who want to peek at foundational deep learning mechanics.

## What To Expect
- Incomplete features and rough edges—this is not meant for production use.
- Some modules may be missing pieces or temporarily broken while I iterate.
- Gradually improving documentation and tests as I learn and refine ideas.

## Contributing / Feedback
If you are exploring similar ideas or spot something that can be improved, feel free to open an issue or start a discussion. Collaboration is welcome, but please keep in mind that the goal here is educational experimentation rather than building a fully featured library.

## Getting Started
- Install dependencies listed in `requirements.txt`.
- Browse `nn/` to see how tensors and operations are currently implemented.
- Run the unit tests to check the latest progress:
  ```
  python -m unittest discover tests
  ```

## Roadmap
- Flesh out automatic differentiation for more operations.
- Add simple neural network layers and loss functions.
- Improve numerical stability and error handling.
- Expand documentation with walkthroughs of key components.

## License
MIT—feel free to fork, experiment, and adapt for your own learning journey.

