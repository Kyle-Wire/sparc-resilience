interface CubeLogoProps {
  size?: number;
  className?: string;
}

export default function CubeLogo({ size = 120, className }: CubeLogoProps) {
  return (
    <img
      src="/logo.svg"
      alt="SPARC Labs"
      width={size}
      height={size}
      className={className}
      style={{ display: "block" }}
    />
  );
}
