int voice_pin = 8;
int LED_pin = 4;
int value;
void setup() {
  // put your setup code here, to run once:
  pinMode(voice_pin,INPUT);
  pinMode(LED_pin,OUTPUT);
}
void loop() {
  digitalWrite(LED_pin, LOW);
  value = digitalRead(voice_pin);
  if(value==HIGH)
   digitalWrite(LED_pin, HIGH); 
}
