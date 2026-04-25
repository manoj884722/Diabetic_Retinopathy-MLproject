import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

function gaussian(x, y, cx, cy, sigma) {
  return Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2));
}

function Retina({ lesions, severity }) {
  const geometry = useMemo(() => {
    const size = 64;
    const geom = new THREE.PlaneGeometry(4, 4, size - 1, size - 1);
    const vertices = geom.attributes.position.array;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const x = (i / size) * 224;
        const y = (j / size) * 224;

        let height = 0;

        lesions.forEach((l) => {
          height += gaussian(x, y, l.x, l.y, l.radius) * 0.6;
        });

        const index = i * size + j;
        vertices[index * 3 + 2] = height;
      }
    }

    geom.computeVertexNormals();
    return geom;
  }, [lesions]);

  let color = "green";
  if (severity === "Mild") color = "yellow";
  if (severity === "Moderate") color = "orange";
  if (severity === "Severe") color = "red";
  if (severity === "Proliferative DR") color = "darkred";

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial color={color} roughness={0.4} metalness={0.1} />
    </mesh>
  );
}

export default function Retina3D({ severity, lesions }) {
  return (
    <div style={{ width: "100%", height: "100%", minHeight: "400px", display: "flex", justifyContent: "center", alignItems: "center" }}>
      <Canvas camera={{ position: [0, 3, 5] }}>
        <ambientLight intensity={1.2} />
        <directionalLight position={[5, 5, 5]} intensity={2} />
        <Retina severity={severity} lesions={lesions} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}