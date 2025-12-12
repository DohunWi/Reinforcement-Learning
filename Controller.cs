using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class Controller : MonoBehaviour
{
    [Header("이동 설정")]
    public float motorForce = 1500f;
    public float brakeForce = 3000f;
    
    [Header("공기역학 & 안정성 (New)")]
    public float downForceValue = 50.0f;   
    public float antiRollVal = 5000.0f;     

    [Header("스티어링 설정")]
    public float maxSteerAngle = 45f;

    [Header("Wheel Colliders")]
    public WheelCollider frontLeftWheel;
    public WheelCollider frontRightWheel;
    public WheelCollider rearLeftWheel;
    public WheelCollider rearRightWheel;

    [Header("Wheel Meshes")]
    public Transform frontLeftMesh;
    public Transform frontRightMesh;
    public Transform rearLeftMesh;
    public Transform rearRightMesh;

    [Header("충돌 및 전복 설정")]
    public float flipTime = 1f;

    [Space]
    public UnityEvent OnCarFlip;

    private float currentFlipTime = 0f;
    private bool isDead = false;
    private Rigidbody rb;

    [HideInInspector] public float steerInput; 
    [HideInInspector] public float motorInput;

    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        
        rb.centerOfMass = new Vector3(0, -0.3f, 0);
    }

    private void FixedUpdate()
    {
        if (isDead) return;

        if (rb.IsSleeping()) rb.WakeUp();

        HandleMotor();
        HandleSteering();
        UpdateWheels();
        
  
        ApplyAntiRoll();
        AddDownForce();
        
        CheckFlip(); 
    }


    void ApplyAntiRoll()
    {
        ApplyAntiRollForce(frontLeftWheel, frontRightWheel);
      
        ApplyAntiRollForce(rearLeftWheel, rearRightWheel);
    }

    void ApplyAntiRollForce(WheelCollider wheelL, WheelCollider wheelR)
    {
        WheelHit hit;
        float travelL = 1.0f;
        float travelR = 1.0f;


        bool groundedL = wheelL.GetGroundHit(out hit);
        if (groundedL)
            travelL = (-wheelL.transform.InverseTransformPoint(hit.point).y - wheelL.radius) / wheelL.suspensionDistance;

   
        bool groundedR = wheelR.GetGroundHit(out hit);
        if (groundedR)
            travelR = (-wheelR.transform.InverseTransformPoint(hit.point).y - wheelR.radius) / wheelR.suspensionDistance;

  
        float antiRollForce = (travelL - travelR) * antiRollVal;

    
        if (groundedL)
            rb.AddForceAtPosition(wheelL.transform.up * -antiRollForce, wheelL.transform.position);
        if (groundedR)
            rb.AddForceAtPosition(wheelR.transform.up * antiRollForce, wheelR.transform.position);
    }

    void AddDownForce()
    {
        rb.AddForce(-transform.up * downForceValue * rb.velocity.magnitude);
    }

    void HandleMotor()
    {
        frontLeftWheel.brakeTorque = 0;
        frontRightWheel.brakeTorque = 0;
        rearLeftWheel.brakeTorque = 0;
        rearRightWheel.brakeTorque = 0;

        frontLeftWheel.motorTorque = motorInput * motorForce;
        frontRightWheel.motorTorque = motorInput * motorForce;
    }

    void HandleSteering()
    {
        float steerAngle = steerInput * maxSteerAngle;
        frontLeftWheel.steerAngle = steerAngle;
        frontRightWheel.steerAngle = steerAngle;
    }

    void UpdateWheels()
    {
        UpdateWheelPose(frontLeftWheel, frontLeftMesh);
        UpdateWheelPose(frontRightWheel, frontRightMesh);
        UpdateWheelPose(rearLeftWheel, rearLeftMesh);
        UpdateWheelPose(rearRightWheel, rearRightMesh);
    }

    void UpdateWheelPose(WheelCollider _collider, Transform _transform)
    {
        Vector3 _pos;
        Quaternion _quat;
        _collider.GetWorldPose(out _pos, out _quat);
        _transform.position = _pos;
        _transform.rotation = _quat;
    }

    void CheckFlip()
    {
        if (Vector3.Dot(transform.up, Vector3.up) < 0.5f)
        {
            currentFlipTime += Time.deltaTime;
        }
        else
        {
            currentFlipTime = 0f;
        }

        if (currentFlipTime > flipTime)
        {
            isDead = true;
            OnCarFlip.Invoke();
        }
    }

    public void ResetCar()
    {
        isDead = false;
        currentFlipTime = 0f;
        steerInput = 0f;
        motorInput = 0f;
        
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        
        frontLeftWheel.motorTorque = 0;
        frontRightWheel.motorTorque = 0;
        
        frontLeftWheel.brakeTorque = float.MaxValue;
        frontRightWheel.brakeTorque = float.MaxValue;
        rearLeftWheel.brakeTorque = float.MaxValue;
        rearRightWheel.brakeTorque = float.MaxValue;
    }
}