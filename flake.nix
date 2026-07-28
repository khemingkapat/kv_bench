{
  description = "KV Cache Optimization & Benchmarking Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "kv-bench-shell";

          buildInputs = with pkgs; [
            python311
            uv
            git
            zlib
            zsh
            stdenv.cc.cc.lib
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
            
            # Ensure local .venv exists and activate it
            if [ ! -d ".venv" ]; then
              echo "Creating isolated uv environment in .venv..."
              uv venv .venv --python python3.11
            fi
            source .venv/bin/activate
            echo "🚀 KV Cache Benchmarking Shell Initialized (.venv active)"

            export SHELL="${pkgs.zsh}/bin/zsh"
            exec "${pkgs.zsh}/bin/zsh"
          '';
        };
      });
}
