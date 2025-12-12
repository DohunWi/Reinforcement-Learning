using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class CarAgent : Agent
{
    public Controller controller; 
    public Transform targetsParent; 
    public float radius = 50.0f;
    private List<Transform> allTargets = new List<Transform>();

    private Rigidbody rBody;
    private Vector3 startPosition;
    private Quaternion startRotation;
    private Transform currentTargetTransform = null; // 현재 쫓고 있는 타깃
    private float previousDistance = 0f;             // 거리 변화량 계산용

    private Vector3 lastPositionCheck = Vector3.zero; // 3초 전 위치 저장
    private float stopTimer = 0f;                     // 시간 재기용

    private bool inPanicMode = false; // 비상 탈출 모드
    private float panicTimer = 0f;    // 탈출 지속 시간
    private float panicSteerDir = 0f;

    void Start()
    {
        Time.timeScale = 1.5f;
    }

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
        startPosition = transform.position;
        startRotation = transform.rotation;

        if (controller == null) controller = GetComponent<Controller>();

        foreach (Transform t in targetsParent)
        {
            allTargets.Add(t);
        }
    }

    public override void OnEpisodeBegin()
    {
        controller.ResetCar();
        
        // 1. 자동차 위치 랜덤
        transform.position = GetRandomPosition(new Vector3(0, 0, 0), radius); 
        transform.rotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);

        // 2. 타깃들 위치 랜덤 및 켜기
        foreach (Transform t in allTargets)
        {
            t.gameObject.SetActive(true);
            t.localPosition = GetRandomPosition(new Vector3(0, 0, 0), radius);
        }

        lastPositionCheck = transform.position;
        stopTimer = 0f;
    }

    // 랜덤 위치 뽑기
    Vector3 GetRandomPosition(Vector3 center, float range)
    {
        int attempts = 0;
        Vector3 potentialPos = Vector3.zero;
        float safetyMargin = 4.0f;

        while (attempts < 100) 
        {
            attempts++;
            float randomX = Random.Range(-range, range);
            float randomZ = Random.Range(-range, range);
            
            potentialPos = center + new Vector3(randomX, 0, randomZ);

            // 장애물(Obstacle)과 겹치는지 확인
            if (!Physics.CheckSphere(potentialPos, safetyMargin, LayerMask.GetMask("Obstacle"))) 
            {
                return new Vector3(potentialPos.x, 0.5f, potentialPos.z);
            }
        }
        // 안전빵
        return new Vector3(center.x, 0.5f, center.z);
    }

    // 가까운 타깃 찾기
    Transform GetClosestTarget()
    {
        Transform closest = null;
        float minDistance = float.MaxValue;
        
        foreach (Transform t in allTargets)
        {
            if (!t.gameObject.activeSelf) continue;

            float dist = Vector3.Distance(transform.position, t.position);
            if (dist < minDistance)
            {
                minDistance = dist;
                closest = t;
            }
        }
        return closest;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Transform closestTarget = GetClosestTarget();

        if (closestTarget != null)
        {
            Vector3 dirToTarget = closestTarget.position - transform.position;
            Vector3 localDir = transform.InverseTransformDirection(dirToTarget.normalized);
            
            sensor.AddObservation(localDir);
            sensor.AddObservation(dirToTarget.magnitude);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(0f);
        }
        sensor.AddObservation(transform.InverseTransformDirection(rBody.velocity));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // 신경망 행동 결정값 
        float steerAction = actions.ContinuousActions[0];
        float motorAction = actions.ContinuousActions[1];

        // 패닉 모드
        if (!inPanicMode && stopTimer > 2.0f)
        {
            float distanceCheck = Vector3.Distance(transform.position, lastPositionCheck);
            if (distanceCheck < 1.5f) 
            {
                inPanicMode = true;       
                panicTimer = 2.0f;        // 2초 비상 탈출 시도
                AddReward(-5.0f);         // 갇힌 것에 대한 벌점

                // 탈출 방향 강제
                // 왼쪽 뒤(-45도), 오른쪽 뒤(+45도)
                Vector3 leftRearDir = (transform.forward * -1 - transform.right).normalized;
                Vector3 rightRearDir = (transform.forward * -1 + transform.right).normalized;

                bool leftBlocked = Physics.Raycast(transform.position, leftRearDir, 3.0f, LayerMask.GetMask("Obstacle"));
                bool rightBlocked = Physics.Raycast(transform.position, rightRearDir, 3.0f, LayerMask.GetMask("Obstacle"));

                // 강제 고정, 탈출 우선
                if (!leftBlocked) panicSteerDir = -1.0f;      
                else if (!rightBlocked) panicSteerDir = 1.0f; 
                else panicSteerDir = 1.0f;                    
            }
            lastPositionCheck = transform.position;
            stopTimer = 0f;
        }

        if (inPanicMode)
        {
            // 장애물 탈출, 강제 조작
            panicTimer -= Time.fixedDeltaTime;

            // 신경망 무시, 강제로 후진
            controller.motorInput = -1.0f;       
            controller.steerInput = panicSteerDir; 

            // 신경망이 이 상황을 배우도록 유도하는 보상
            // 우연히라도 후진하려 했다면 보상
            if (motorAction < 0) AddReward(0.05f);
            // 방향을 맞게 꺾으려 했다면 보상
            if (Mathf.Sign(steerAction) == Mathf.Sign(panicSteerDir)) AddReward(0.05f);

            if (panicTimer <= 0f) inPanicMode = false;
        }
        else
        {
            // 자율 주행, 일반 상황
            controller.steerInput = steerAction;
            controller.motorInput = motorAction;
            
            // 앞이 막혔는지 확인
            bool isFrontBlocked = Physics.SphereCast(transform.position + Vector3.up * 0.5f, 0.5f, transform.forward, out RaycastHit hit, 2.0f, LayerMask.GetMask("Obstacle"));

            // 타깃 정보, 가려짐 확인 
            Transform closestTarget = GetClosestTarget();
            bool isTargetObstructed = false;
            
            if (closestTarget != null)
            {
                if (Physics.Linecast(transform.position + Vector3.up * 0.5f, closestTarget.position + Vector3.up * 0.5f, LayerMask.GetMask("Obstacle")))
                {
                    isTargetObstructed = true;
                }
            }

            // 타깃 변경 감지
            if (closestTarget != currentTargetTransform)
            {
                currentTargetTransform = closestTarget;
                if (closestTarget != null) previousDistance = Vector3.Distance(transform.position, closestTarget.position);
            }

            // 속도 성분 분리
            Vector3 localVel = transform.InverseTransformDirection(rBody.velocity);
            float forwardSpeed = localVel.z;
            float turnSpeed = Mathf.Abs(rBody.angularVelocity.y);

            // 보상 로직 
            // 제자리 뺑뺑이 방지 규칙
            if (Physics.Raycast(transform.position, transform.forward, out RaycastHit longHit, 10.0f, LayerMask.GetMask("Obstacle")))
            {
                float distanceToWall = longHit.distance;
                float currentSpeed = rBody.velocity.magnitude; 

                // 벽 근처 가면 감속 유도
                float safeSpeed = distanceToWall * 1.0f; 
                if (currentSpeed > safeSpeed)
                {
                    // 초과한 속도만큼 비례해서 처벌
                    float excessSpeed = currentSpeed - safeSpeed;
                    AddReward(-0.05f * excessSpeed); 
                }
            }
            if (forwardSpeed < 1.0f && turnSpeed > 3.0f)
            {
                AddReward(-0.1f);
            }

            // 막혔을 때
            if (isFrontBlocked)
            {
                // 후진 유도
                if (motorAction < -0.1f) 
                {
                    float reward = 0.0f; 
                    if (Mathf.Abs(steerAction) > 0.5f) reward += 0.01f; 
                    AddReward(reward);
                }
                else 
                {
                    AddReward(-0.1f); // 벽 보고 멍때리면 처벌
                }
            }
            else
            {
                // 앞이 뚫림 -> 정상 주행
                if (currentTargetTransform != null)
                {
                    float currentDistance = Vector3.Distance(transform.position, currentTargetTransform.position);
                    float distanceDelta = previousDistance - currentDistance;

                    if (isTargetObstructed)
                    {
                        // 벽 뒤 타깃은 거리 무시하고 달리면 보상 -> 탐험 유도 하기
                        if (forwardSpeed > 2.0f) AddReward(0.01f);
                        else AddReward(-0.001f);
                    }
                    else
                    {
                        // 타깃이 보이면 거리 기반 보상
                        if (distanceDelta > 0) AddReward(distanceDelta * 1.0f);
                        else AddReward(distanceDelta * 1.5f); // 멀어지면 벌점
                    }
                    previousDistance = currentDistance;
                }

                // 멀쩡한 길에서 후진하면 벌점 (Moonwalking 방지)
                if (motorAction < 0) AddReward(-0.05f);
            }
        }
        stopTimer += Time.fixedDeltaTime;
        
        // 시간 페널티
        AddReward(-0.005f);
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Horizontal"); 
        continuousActions[1] = Input.GetAxis("Vertical");   
    }
    
    // 충돌 처리
    private void OnTriggerEnter(Collider other)
    {
        // 타깃을 먹었을 때
        if (other.CompareTag("Target"))
        {
            AddReward(15.0f);
            
            // 타깃 비활성화
            other.gameObject.SetActive(false);
            if (GetClosestTarget() == null)
            {
                // 다 먹었으면 에피소드 종료
                EndEpisode();
            }
        }
    }

    // 벽 충돌 처리
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle")) 
        {
            // 충돌 강도 측정
            float impactForce = collision.relativeVelocity.magnitude;

            // 기본 벌점 + 속도 비례 벌점
            float totalPenalty = -1.0f - (impactForce * 0.1f);

            SetReward(totalPenalty);
            
            Debug.Log($"쾅! 충돌 속도: {impactForce:F1} | 받은 벌점: {totalPenalty:F1}");

            EndEpisode();
        }
    }
}