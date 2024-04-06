// ReadCDB_VCDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ReadCDB_VC.h"
#include "ReadCDB_VCDlg.h"
#include ".\readcdb_vcdlg.h"
#include <atlimage.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About
struct CSample
{
	unsigned char w;
	unsigned char h;
	unsigned char lable;
	BYTE* Data;	
};

CSample *Samples;
byte imgType;
int level, idx = 0;
CImage img;

unsigned long CurAddr = 0;
void ReadData(CString FileName);
void ReadRamFile(void* Buffer, int ByteCount);
enum ImageType{itBinary=0, itGray=1};

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

	// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CReadCDB_VCDlg dialog



CReadCDB_VCDlg::CReadCDB_VCDlg(CWnd* pParent /*=NULL*/)
: CDialog(CReadCDB_VCDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CReadCDB_VCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CReadCDB_VCDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BTN_READ, OnBnClickedBtnRead)
	ON_BN_CLICKED(IDC_BTN_NEXT, OnBnClickedBtnNext)
	ON_BN_CLICKED(IDC_BTN_PREV, OnBnClickedBtnPrev)
END_MESSAGE_MAP()


// CReadCDB_VCDlg message handlers

BOOL CReadCDB_VCDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CReadCDB_VCDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CReadCDB_VCDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CReadCDB_VCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CReadCDB_VCDlg::OnBnClickedBtnRead()
{
	CFileDialog fdlg(true, ".cdb", 0, 0, "CDB Files (*.cdb)|*.cdb|All Files(*.*)|*.*|");
	if(fdlg.DoModal() == IDOK)
		ReadData(fdlg.GetFileName());
	OnBnClickedBtnNext();
	//SetCurrentDirectory("H:\\OCR\\DataBase\\New Digit DB\\All in three part\\");
}

//imgType = 0 for binary and 1 for grayscale file format
void ReadData(CString FileName)
{
	HANDLE hFile, hMap;
	void* pBase;
	BYTE d,m,W,H,x,y,StartByte,counter,WBcount;
	WORD yy, ByteCount;
	DWORD* LetterCount;
	char Comments[256];
	DWORD TotalRec;
	bool normal,bWhite;
	int i;

	hFile = CreateFile(FileName, GENERIC_READ,
		FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, 0);
	if(hFile == NULL)
	{
		hFile = 0;
		MessageBox(HWND(NULL), "File Can not be loaded", "error", MB_ICONERROR);
		return;
	};

	hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	pBase = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(hMap);
	hMap = 0;
	CurAddr = unsigned long(pBase);

	//read private header
	ReadRamFile(&yy, 2);
	ReadRamFile(&m, 1);
	ReadRamFile(&d, 1);
	ReadRamFile(&W, 1);
	ReadRamFile(&H, 1);
	ReadRamFile(&TotalRec, 4);

	LetterCount = new DWORD[128];
	ZeroMemory(LetterCount, 128*sizeof(DWORD));

	ReadRamFile(LetterCount, 4*128);
	ReadRamFile(&imgType, 1);
	if(imgType != itGray)
		imgType = itBinary;
	ReadRamFile(Comments, 256);
	if( (W > 0) && (H > 0))
		normal = true;
	else
		normal = false;

	CurAddr = unsigned long(pBase) + 1024;//bypass 1024 bytes header
	Samples = new CSample[TotalRec];
	for (i = 0; i < TotalRec; i++) 
	{
		ReadRamFile(&StartByte, 1);//must be 0xff
		ReadRamFile(&Samples[i].lable, 1);
		if (!normal) 
		{
			ReadRamFile(&W, 1);
			ReadRamFile(&H, 1);
		};
		ReadRamFile(&ByteCount, 2);
		Samples[i].h = H;
		Samples[i].w = W;
		Samples[i].Data = new BYTE[H*W];

		if(imgType == itBinary)
		{
			for (y = 0; y < H; y++)
			{
				bWhite = true;
				counter = 0;
				while (counter < W)
				{
					ReadRamFile(&WBcount, 1);
					x = 0;
					while(x < WBcount)
					{
						if(bWhite)
							Samples[i].Data[y*W + x+counter] = 0;//Background
						else
							Samples[i].Data[y*W + x+counter]= 1;//ForeGround
						x++;
					};
					bWhite = !bWhite;//black white black white ...
					counter += WBcount;
				}
			};
		}
		else//GrayScale mode
		{
			ReadRamFile(Samples[i].Data, W*H);
		};
	};//i
	if(imgType == itGray)
		level = 1;
	else
		level = 255;
	UnmapViewOfFile(pBase);
	CloseHandle(hFile); 
}

void ReadRamFile(void* Buffer, int ByteCount)
{
	CopyMemory(Buffer, (void*)CurAddr, ByteCount);
	CurAddr += ByteCount;
};

void CReadCDB_VCDlg::OnBnClickedBtnNext()
{
	RedrawWindow();
	int w = Samples[idx].w;
	int h = Samples[idx].h;	
	int x,y;
	byte* b;
	if(!img.IsNull())
		img.Detach();
	img.Create(w,h,24);
	for (y = 0; y < h; y++)
	{
		b = (byte*)img.GetPixelAddress(0,y);
		for (x = 0; x < w; x++)
		{
			*b++ = Samples[idx].Data[y*w+x]*level;
			*b++ = Samples[idx].Data[y*w+x]*level;
			*b++ = Samples[idx].Data[y*w+x]*level;
		}
	}	
	img.Draw(GetDC()->m_hDC, 40,40,w,h,0,0,w,h);
	idx++;
	CString str;
	str.Format("%d.   [Code: %d]    [W: %d]    [H: %d]", idx, Samples[idx-1].lable, w, h);
	SetDlgItemText(IDC_LBL, str);
}

void CReadCDB_VCDlg::OnBnClickedBtnPrev()
{
	idx = max(0, idx-2);
	OnBnClickedBtnNext();
}
