export default function Placeholder({ name }: { name: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <h2 className="text-xl font-bold">{name}</h2>
        <p className="mt-2 text-sm text-sparc-gray-600">Coming in Phase 2/3</p>
      </div>
    </div>
  );
}
