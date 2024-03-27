#include <Ultrasonic.h>
#include <Servo.h>
#define in1 5
#define in2 11
#define in3 6
#define in4 3
#define servo_port 9
#define pwm 63
#define limit 20
#define comp_Front 7
#define comp_Rear 4
#define vth_Batt 7400
#define mid 99
Ultrasonic ultra_Front(18, 17);
Ultrasonic ultra_Rear(16, 15);
String inputString = "";
boolean stringComplete = false;
int distance_Front, distance_Rear, i;
long volt_Batt;
Servo myservo;

void motor_F()
{
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  myservo.write(mid);
}

void motor_B()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  myservo.write(mid);
}

void motor_L()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  myservo.write(mid - 30);
}

void motor_R()
{
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  myservo.write(mid + 30);
}

void motor_S()
{
  digitalWrite(in1, HIGH);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, HIGH);
  myservo.write(mid);
}

void motor_I()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  myservo.write(mid);
}

void motor_FL()
{
  analogWrite(in1, pwm);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  myservo.write(mid - 30);
}

void motor_FR()
{
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  analogWrite(in3, pwm);
  digitalWrite(in4, LOW);
  myservo.write(mid + 30);
}

void motor_BL()
{
  digitalWrite(in1, LOW);
  analogWrite(in2, pwm);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  myservo.write(mid - 30);
}

void motor_BR()
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  analogWrite(in4, pwm);
  myservo.write(mid + 30);
}

void setup() {
  // put your setup code here, to run once:
  myservo.attach(servo_port);
  Serial.begin(38400);
  inputString.reserve(200);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  motor_I();
}

void loop() {
  // put your main code here, to run repeatedly:
  if(stringComplete)
  {
    distance_Front = ultra_Front.getDistanceInCM() - comp_Front;
    distance_Rear = ultra_Rear.getDistanceInCM() - comp_Rear;
    if((distance_Front < limit) && (distance_Rear < limit))
    {
      Serial.println("Close to BOTH Obstacles!");
      Serial.print("Front: ");
      Serial.print(distance_Front);
      Serial.print("cm. Rear: ");
      Serial.print(distance_Rear);
      Serial.println("cm.");
      inputString = "Stop";
      delay(20);
    }
    else if(distance_Front < limit)
    {
      Serial.println("Close to Front Obstacles. AUTO-BACK.");
      Serial.print("Front: ");
      Serial.print(distance_Front);
      Serial.println("cm.");
      inputString = "Back";
      delay(20);
    }
    else if(distance_Rear < limit)
    {
      Serial.println("Close to Rear Obstacles. AUTO-FORWARD.");
      Serial.print("Rear: ");
      Serial.print(distance_Rear);
      Serial.println("cm.");
      inputString = "Forward";
      delay(20);
    }
    volt_Batt = 0;
    for(i = 0; i < 100; i++)
    {
      volt_Batt += long(analogRead(A0));
      delayMicroseconds(100);
    }
    volt_Batt = volt_Batt * 50 * 11 / 1024;
    if(volt_Batt < vth_Batt)
    {
      Serial.print("Low Battery! Voltage: ");
      Serial.print(volt_Batt);
      Serial.println("mV.");
      inputString = "Idle";
      delay(20);
    }
    if(inputString != "Idle")
    {
      Serial.println(inputString);
    }
    if(inputString == "Forward")
    {
      motor_F();
    }
    else if(inputString == "Back")
    {
      motor_B();
    }
    else if(inputString == "Left")
    {
      motor_L();
    }
    else if(inputString == "Right")
    {
      motor_R();
    }
    else if(inputString == "Stop")
    {
      motor_S();
    }
    else if(inputString == "Idle")
    {
      motor_I();
    }
    else if(inputString == "Forward Left")
    {
      motor_FL();
    }
    else if(inputString == "Forward Right")
    {
      motor_FR();
    }
    else if(inputString == "Back Left")
    {
      motor_BL();
    }
    else if(inputString == "Back Right")
    {
      motor_BR();
    }
    else if(inputString == "Read")
    {
      Serial.print("Front: ");
      Serial.print(distance_Front);
      Serial.print("cm. Rear: ");
      Serial.print(distance_Rear);
      Serial.println("cm.");
      Serial.print("Battery: ");
      Serial.print(volt_Batt);
      Serial.println("mV.");
      motor_I();
    }
    else
    {
      Serial.println("Invalid Command!");
    }
    inputString = "";
    stringComplete = false;
    delay(10);
  }
}

void serialEvent()
{
  while(Serial.available())
  {
    char inChar = (char)Serial.read();
    if((inChar != '\r') && (inChar != '\n'))
    {
      inputString += inChar;
    }
    if(inChar == '\n')
    {
      stringComplete = true;
    }
  }
}