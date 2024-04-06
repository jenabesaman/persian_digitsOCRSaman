object Form1: TForm1
  Left = 192
  Top = 114
  Width = 218
  Height = 237
  Caption = 'Read CDB File'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  PixelsPerInch = 96
  TextHeight = 13
  object Image1: TImage
    Left = 56
    Top = 32
    Width = 113
    Height = 113
    Proportional = True
  end
  object Label1: TLabel
    Left = 80
    Top = 128
    Width = 52
    Height = 24
    Alignment = taCenter
    Caption = 'Label'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clRed
    Font.Height = -19
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
    Visible = False
  end
  object btnNext: TButton
    Left = 144
    Top = 160
    Width = 50
    Height = 25
    Caption = 'next -->'
    Enabled = False
    TabOrder = 0
    OnClick = btnNextClick
  end
  object btnPrev: TButton
    Left = 80
    Top = 160
    Width = 50
    Height = 25
    Caption = '<-- Prev'
    Enabled = False
    TabOrder = 1
    OnClick = btnPrevClick
  end
  object btnRead: TBitBtn
    Left = 16
    Top = 160
    Width = 49
    Height = 25
    Caption = 'Read'
    Default = True
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clGreen
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = [fsBold]
    ParentFont = False
    TabOrder = 2
    OnClick = btnReadClick
  end
  object OpenDialog1: TOpenDialog
    Filter = 'Character DataBase File|*.cdb|All Files|*.*'
    Left = 192
    Top = 112
  end
end
