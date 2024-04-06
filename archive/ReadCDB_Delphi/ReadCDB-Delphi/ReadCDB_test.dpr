program ReadCDB_test;

uses
  Forms,
  Main in 'Main.pas' {Form1},
  ReadCDB in 'ReadCDB.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
