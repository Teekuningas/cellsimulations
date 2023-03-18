{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/22.11";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in {
        formatter = pkgs.nixfmt;
        devShells.default = pkgs.mkShell {
          packages =
            [ (pkgs.python39.withPackages (ps: [ ps.matplotlib ps.black ])) ];
        };
      });
}
