unit Main;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, ReadCDB, StdCtrls, ExtCtrls, Buttons;

type
  TForm1 = class(TForm)
    btnNext: TButton;
    Image1: TImage;
    Label1: TLabel;
    OpenDialog1: TOpenDialog;
    btnPrev: TButton;
    btnRead: TBitBtn;
    procedure btnNextClick(Sender: TObject);
    procedure btnReadClick(Sender: TObject);
    procedure btnPrevClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;
  data: Tdata;
  labels: TLabels;
  idx: integer = 0;
  imgType: TImageType;

implementation

{$R *.dfm}

procedure TForm1.btnNextClick(Sender: TObject);
var
   w,h,x,y,level: integer;
begin
   w := Length(data[idx][0]);
   h := Length(data[idx]);
   Image1.picture.bitmap.Width := w;
   Image1.picture.bitmap.Height := h;
   if(imgType = itGray) then
      level := 1
   else
      level := 255;
   for y := 0 to h-1 do
      for x := 0 to w-1 do
         image1.Picture.Bitmap.Canvas.Pixels[x,y] := RGB(data[idx][y,x]*level, data[idx][y,x]*level, data[idx][y,x]*level);

   Label1.Caption := IntToStr(labels[idx]);
   idx := idx + 1;
end;

procedure TForm1.btnReadClick(Sender: TObject);
begin
   if (OpenDialog1.Execute) then
   begin
      Label1.Visible := true;
      Label1.Caption := 'Please Wait...';
      Refresh;
      ReadData(OpenDialog1.FileName, data, labels, imgType);
      Label1.Caption := '';
      btnNextClick(self);
      btnNext.Enabled := true;
      btnPrev.Enabled := true;
   end;
end;

procedure TForm1.btnPrevClick(Sender: TObject);
begin
   if(idx > 1) then
   begin
      idx := idx - 2;
      btnNextClick(self);
   end;   
end;

end.
