import React from 'react';
import styled, { keyframes } from 'styled-components';

const SkeletonContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const SkeletonCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const SkeletonHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const loading = keyframes`
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
`;

const SkeletonAvatar = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
`;

const SkeletonInfo = styled.div`
  flex: 1;
`;

const SkeletonName = styled.div`
  height: 20px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 8px;
  width: 60%;
`;

const SkeletonTitle = styled.div`
  height: 16px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 4px;
  width: 40%;
`;

const SkeletonScore = styled.div`
  width: 80px;
  height: 32px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 16px;
`;

const SkeletonContent = styled.div`
  margin-bottom: 1rem;
`;

const SkeletonContact = styled.div`
  height: 16px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 8px;
  width: 80%;
`;

const SkeletonSkills = styled.div`
  display: flex;
  gap: 8px;
  margin-bottom: 1rem;
`;

const SkeletonSkill = styled.div`
  height: 24px;
  width: 60px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 12px;
`;

const SkeletonFooter = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const SkeletonButton = styled.div`
  height: 32px;
  width: 100px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 16px;
`;

const SkeletonIcon = styled.div`
  height: 20px;
  width: 20px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${loading} 1.5s infinite;
  border-radius: 4px;
`;

function LoadingSkeleton({ count = 6 }) {
  return (
    <SkeletonContainer>
      {Array.from({ length: count }, (_, index) => (
        <SkeletonCard key={index}>
          <SkeletonHeader>
            <SkeletonAvatar />
            <SkeletonInfo>
              <SkeletonName />
              <SkeletonTitle />
            </SkeletonInfo>
            <SkeletonScore />
          </SkeletonHeader>
          
          <SkeletonContent>
            <SkeletonContact />
            <SkeletonContact style={{ width: '60%' }} />
            <SkeletonContact style={{ width: '70%' }} />
          </SkeletonContent>
          
          <SkeletonSkills>
            <SkeletonSkill />
            <SkeletonSkill />
            <SkeletonSkill />
            <SkeletonSkill />
          </SkeletonSkills>
          
          <SkeletonFooter>
            <SkeletonButton />
            <SkeletonIcon />
          </SkeletonFooter>
        </SkeletonCard>
      ))}
    </SkeletonContainer>
  );
}

export default LoadingSkeleton;
