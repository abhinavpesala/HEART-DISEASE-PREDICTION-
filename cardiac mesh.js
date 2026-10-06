import { useRef } from "react";
import { useFrame } from "@react-three/fiber";

export default function CardiacMesh({ risk }) {
  const meshRef = useRef();

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  // Color code: Red (high risk), Yellow (moderate), Green (low)
  const getColor = () => {
    if (risk >= 0.7) return "#ff4d4d";
    if (risk >= 0.4) return "#ffb84d";
    return "#42d392";
  };

  return (
    <group ref={meshRef}>
      {/* Main heart body */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshStandardMaterial color={getColor()} transparent opacity={0.8} />
      </mesh>

      {/* Left ventricle (LAD) */}
      <mesh position={[0.8, -0.3, 0.1]}>
        <sphereGeometry args={[0.4, 20, 20]} />
        <meshStandardMaterial color="#4aa3ff" />
      </mesh>

      {/* Right ventricle (RCA) */}
      <mesh position={[-0.9, -0.2, 0.2]}>
        <sphereGeometry args={[0.35, 20, 20]} />
        <meshStandardMaterial color="#4aa3ff" />
      </mesh>

      {/* Atria (LCX) */}
      <mesh position={[0, 0.8, 0]}>
        <sphereGeometry args={[0.3, 20, 20]} />
        <meshStandardMaterial color="#4aa3ff" />
      </mesh>
    </group>
  );
}