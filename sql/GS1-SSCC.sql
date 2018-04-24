/****** Object:  StoredProcedure [dbo].[SEQ_NEXT_ID]    Script Date: 24/04/2018 16:00:00 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Matthieu
-- Create date: 28-02-2018
-- Description:	Return an SSCC
-- =============================================
ALTER PROCEDURE [dbo].[SEQ_NEXT_ID] 
@PREFIX NVARCHAR(8)
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    --PREFIX = [1][780][465287] => [1 - 5][COUNTRY CODE][COMPANY CODE]
	  SELECT  FORMAT((NEXT VALUE FOR dbo.INCREMENTS_PALLET),CONCAT(@PREFIX, '##########')) AS Id;  
END
