# Tangles-Rust
## How to install
First, install maturin. This is used to provide the Rust module to python.
Next, in this folder, run `maturin build`. You will find the wheels under the `target` folder. In your desired conda environment, you can now run `pip install target/<name of wheel>` to install the package. You can the import it as `import tangles_rust`from whereever you want.