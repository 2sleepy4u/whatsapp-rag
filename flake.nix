{
	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
		opencode.url = "github:anomalyco/opencode/v1.18.18"; # tag, branch, or commit
	};

	outputs = { self, nixpkgs, opencode }:
		let
		system = "x86_64-linux";
	pkgs = nixpkgs.legacyPackages.${system};
	in
	{
		devShells.${system}.default = pkgs.mkShell {
			buildInputs = [
				pkgs.unzip
				opencode.packages.${system}.default
					pkgs.uv
					pkgs.python3
			];
		};
	};
}
