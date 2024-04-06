// ReadCDB_VCDlg.h : header file
//

#pragma once


// CReadCDB_VCDlg dialog
class CReadCDB_VCDlg : public CDialog
{
// Construction
public:
	CReadCDB_VCDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_READCDB_VC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnRead();
	afx_msg void OnBnClickedBtnNext();
	afx_msg void OnBnClickedBtnPrev();
};
